import os
import sys
import time
from subprocess import CalledProcessError
from typing import Dict, List, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures

from indextts.utils.front import TextNormalizer, TextTokenizer

# Configuration for punctuation marks to remove from TTS input
# These characters can cause unwanted pauses in speech synthesis
TTS_PUNCTUATION_TO_REMOVE = {
    # Dots (except for sentence endings)
    '.': '',  # Will be handled specially to preserve sentence endings
    
    # English quotation marks
    '"': '',
    "'": '',
    '\u201C': '',  # " Left double quotation mark
    '\u201D': '',  # " Right double quotation mark
    '\u2018': '',  # ' Left single quotation mark
    '\u2019': '',  # ' Right single quotation mark
    
    # Chinese quotation marks
    '「': '',
    '」': '',
    '『': '',
    '』': '',
    
    # Chinese book title marks
    '《': '',
    '》': '',
    '〈': '',
    '〉': '',
    
    # Dashes
    '-': '',
    '—': '',
    '–': '',
    '～': '',
    
    # Parentheses and brackets
    '(': '',
    ')': '',
    '[': '',
    ']': '',
    '{': '',
    '}': '',
    '（': '',
    '）': '',
    '【': '',
    '】': '',
    '〔': '',
    '〕': '',
    '［': '',
    '］': '',
    
    # Additional symbols that might affect TTS
    '*': '',
    '#': '',
    '@': '',
    '&': '',
    '\\': '',
    '|': '',
    '^': '',
    '~': '',
    '`': '',
    '·': '',
}

def remove_tts_punctuation(text):
    """
    Removes punctuation marks that can cause unwanted pauses in TTS.
    Preserves basic sentence punctuation (commas, periods, question marks, exclamation marks).
    """
    if not text:
        return text
    
    # Handle dots specially - only remove if not at sentence end
    import re
    
    # First, check if the text ends with a typical sentence ending
    has_sentence_ending = bool(re.search(r'[.!?。！？]\s*$', text))
    
    # Remove dots in abbreviations
    # 1. Dot between letters (e.g., "J.K." -> "JK")
    text = re.sub(r'(?<=[A-Za-z])\.(?=[A-Za-z])', '', text)
    
    # 2. Dot after letter followed by space and uppercase (e.g., "Dr. Smith" -> "Dr Smith")
    text = re.sub(r'(?<=[A-Za-z])\.(?=\s+[A-Z])', '', text)
    
    # 3. Dot at the end after uppercase letters (abbreviations like "U.S.A.")
    # Only if it doesn't look like a sentence ending
    if not has_sentence_ending or not re.search(r'[a-z]\.\s*$', text):
        text = re.sub(r'(?<=[A-Z])\.(?=\s*$)', '', text)
    
    # Remove other punctuation marks
    for char, replacement in TTS_PUNCTUATION_TO_REMOVE.items():
        if char != '.':  # Skip dot as we handled it above
            text = text.replace(char, replacement)
    
    return text


class IndexTTS:
    def __init__(
        self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, device=None, use_cuda_kernel=None,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # Comment-off to load the VQ-VAE model for debugging tokenizer
        #   https://github.com/index-tts/index-tts/issues/34
        #
        # from indextts.vqvae.xtts_dvae import DiscreteVAE
        # self.dvae = DiscreteVAE(**self.cfg.vqvae)
        # self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        # load_checkpoint(self.dvae, self.dvae_path)
        # self.dvae = self.dvae.to(self.device)
        # if self.is_fp16:
        #     self.dvae.eval().half()
        # else:
        #     self.dvae.eval()
        # print(">> vqvae weights restored from:", self.dvae_path)
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        if self.is_fp16:
            try:
                import deepspeed

                use_deepspeed = True
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> DeepSpeed加载失败，回退到标准推理: {e}")
                print("See more details https://www.deepspeed.ai/tutorials/advanced-install/")

            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load as anti_alias_activation_loader
                anti_alias_activation_cuda = anti_alias_activation_loader.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.", e, file=sys.stderr)
                print(" Reinstall with `pip install -e . --no-deps --no-build-isolation` to prebuild `anti_alias_activation_cuda` kernel.", file=sys.stderr)
                print(
                    "See more details: https://github.com/index-tts/index-tts/issues/164#issuecomment-2903453206", file=sys.stderr
                )
                self.use_cuda_kernel = False
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)
        # 缓存参考音频mel：
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        # 进度引用显示（可选）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                first_stop_position = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)
                
                # 计算截断位置的百分比
                truncation_ratio = first_stop_position / code.size(0)
                
                # 检查stop_mel_token是否过早出现
                if truncation_ratio < 0.9:  # 如果在前90%就出现
                    # 抛出异常，让上层处理重试
                    raise RuntimeError(f"Early stop_mel_token detected at {truncation_ratio*100:.1f}% (position {first_stop_position}/{code.size(0)}). This likely indicates truncated speech generation.")
                elif truncation_ratio < 0.95:
                    # 可疑位置，打印警告
                    print(f"[WARNING] stop_mel_token at {truncation_ratio*100:.1f}% - might be early termination")
                
                len_ = first_stop_position

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def bucket_sentences(self, sentences, bucket_max_size=4) -> List[List[Dict]]:
        """
        Sentence data bucketing.
        if ``bucket_max_size=1``, return all sentences in one bucket.
        """
        outputs: List[Dict] = []
        for idx, sent in enumerate(sentences):
            outputs.append({"idx": idx, "sent": sent, "len": len(sent)})
       
        if len(outputs) > bucket_max_size:
            # split sentences into buckets by sentence length
            buckets: List[List[Dict]] = []
            factor = 1.5
            last_bucket = None
            last_bucket_sent_len_median = 0

            for sent in sorted(outputs, key=lambda x: x["len"]):
                current_sent_len = sent["len"]
                if current_sent_len == 0:
                    print(">> skip empty sentence")
                    continue
                if last_bucket is None \
                        or current_sent_len >= int(last_bucket_sent_len_median * factor) \
                        or len(last_bucket) >= bucket_max_size:
                    # new bucket
                    buckets.append([sent])
                    last_bucket = buckets[-1]
                    last_bucket_sent_len_median = current_sent_len
                else:
                    # current bucket can hold more sentences
                    last_bucket.append(sent) # sorted
                    mid = len(last_bucket) // 2
                    last_bucket_sent_len_median = last_bucket[mid]["len"]
            last_bucket=None
            # merge all buckets with size 1
            out_buckets: List[List[Dict]] = []
            only_ones: List[Dict] = []
            for b in buckets:
                if len(b) == 1:
                    only_ones.append(b[0])
                else:
                    out_buckets.append(b)
            if len(only_ones) > 0:
                # merge into previous buckets if possible
                # print("only_ones:", [(o["idx"], o["len"]) for o in only_ones])
                for i in range(len(out_buckets)):
                    b = out_buckets[i]
                    if len(b) < bucket_max_size:
                        b.append(only_ones.pop(0))
                        if len(only_ones) == 0:
                            break
                # combined all remaining sized 1 buckets
                if len(only_ones) > 0:
                    out_buckets.extend([only_ones[i:i+bucket_max_size] for i in range(0, len(only_ones), bucket_max_size)])
            return out_buckets
        return [outputs]

    def pad_tokens_cat(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        if self.model_version and self.model_version >= 1.5:
            # 1.5版本以上，直接使用stop_text_token 右侧填充，填充到最大长度
            # [1, N] -> [N,]
            tokens = [t.squeeze(0) for t in tokens]
            return pad_sequence(tokens, batch_first=True, padding_value=self.cfg.gpt.stop_text_token, padding_side="right")
        max_len = max(t.size(1) for t in tokens)
        outputs = []
        for tensor in tokens:
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                n = min(8, pad_len)
                tensor = torch.nn.functional.pad(tensor, (0, n), value=self.cfg.gpt.stop_text_token)
                tensor = torch.nn.functional.pad(tensor, (0, pad_len - n), value=self.cfg.gpt.start_text_token)
            tensor = tensor[:, :max_len]
            outputs.append(tensor)
        tokens = torch.cat(outputs, dim=0)
        return tokens

    def torch_empty_cache(self):
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception as e:
            pass

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)
    
    def _extract_original_sentences(self, original_text, normalized_sentences):
        """
        Extract original sentences from the original text based on sentence boundaries.
        This preserves all original punctuation and formatting.
        """
        import re
        
        # Split original text by sentence-ending punctuation
        # Include the punctuation in the sentence
        sentence_pattern = r'[^。！？.!?\n]+[。！？.!?\n]?'
        original_parts = re.findall(sentence_pattern, original_text)
        
        # Clean up empty parts and strip whitespace
        original_parts = [part.strip() for part in original_parts if part.strip()]
        
        # If no sentences found, treat the whole text as one sentence
        if not original_parts:
            original_parts = [original_text.strip()]
        
        # Match the number of sentences from normalized tokenization
        # This ensures alignment between TTS processing and SRT
        if len(original_parts) == len(normalized_sentences):
            return original_parts
        elif len(original_parts) > len(normalized_sentences):
            # Merge some sentences
            merged = []
            parts_per_sentence = len(original_parts) // len(normalized_sentences)
            remainder = len(original_parts) % len(normalized_sentences)
            
            idx = 0
            for i in range(len(normalized_sentences)):
                count = parts_per_sentence + (1 if i < remainder else 0)
                merged_sentence = ' '.join(original_parts[idx:idx+count])
                merged.append(merged_sentence)
                idx += count
            
            return merged
        else:
            # We have fewer original parts than normalized sentences
            # This might happen with very long sentences that get split
            # In this case, use the normalized sentences as fallback
            result = []
            for sent_tokens in normalized_sentences:
                sent_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
                sent_text = self.tokenizer.decode(sent_ids).strip()
                result.append(sent_text)
            return result

    # 快速推理：对于“多句长文本”，可实现至少 2~10 倍以上的速度提升~ （First modified by sunnyboxs 2025-04-16）
    def infer_fast(self, audio_prompt, text, output_path, verbose=False, max_text_tokens_per_sentence=300, sentences_bucket_max_size=4, **generation_kwargs):
        """
        Args:
            ``max_text_tokens_per_sentence``: 分句的最大token数，默认``100``，可以根据GPU硬件情况调整
                - 越小，batch 越多，推理速度越*快*，占用内存更多，可能影响质量
                - 越大，batch 越少，推理速度越*慢*，占用内存和质量更接近于非快速推理
            ``sentences_bucket_max_size``: 分句分桶的最大容量，默认``4``，可以根据GPU内存调整
                - 越大，bucket数量越少，batch越多，推理速度越*快*，占用内存更多，可能影响质量
                - 越小，bucket数量越多，batch越少，推理速度越*慢*，占用内存和质量更接近于非快速推理
        """
        print(">> start fast inference...")
        
        self._set_gr_progress(0, "start fast inference...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass

        auto_conditioning = cond_mel
        cond_mel_lengths = torch.tensor([cond_mel_frame], device=self.device)

        # text_tokens
        # Store original text for SRT generation
        original_text = text
        
        # For TTS processing, use normalized tokens
        normalized_text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(normalized_text_tokens_list, max_tokens_per_sentence=max_text_tokens_per_sentence)
        
        # Extract original sentences based on character positions
        # This approach preserves all original punctuation and formatting
        original_sentences = self._extract_original_sentences(original_text, sentences)

        if verbose:
            print(">> text token count:", len(normalized_text_tokens_list))
            print("   splited sentences count:", len(sentences))
            print("   max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        # 动态计算max_mel_tokens，但不能超过模型训练时的限制
        # 模型配置中 max_mel_tokens=605, max_text_tokens=402
        model_max_mel_tokens = self.cfg.gpt.max_mel_tokens  # 605
        default_max_mel = min(model_max_mel_tokens - 5, max(600, max_text_tokens_per_sentence * 3))  # 留5个token余量，避免越界
        
        # 调试信息
        print(f"[DEBUG] model_max_mel_tokens from config: {model_max_mel_tokens}")
        print(f"[DEBUG] max_text_tokens_per_sentence: {max_text_tokens_per_sentence}")
        print(f"[DEBUG] calculated default_max_mel: {default_max_mel}")
        print(f"[DEBUG] generation_kwargs before pop: {generation_kwargs}")
        
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", default_max_mel)
        
        print(f"[DEBUG] max_mel_tokens after pop: {max_mel_tokens}")
        
        # 确保不超过模型限制
        if max_mel_tokens > model_max_mel_tokens - 5:
            print(f"[WARNING] Requested max_mel_tokens ({max_mel_tokens}) exceeds model limit ({model_max_mel_tokens}), capping to {model_max_mel_tokens - 5}")
            max_mel_tokens = model_max_mel_tokens - 5
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        # text processing
        all_text_tokens: List[List[torch.Tensor]] = []
        self._set_gr_progress(0.1, "text processing...")
        bucket_max_size = sentences_bucket_max_size if self.device != "cpu" else 1
        all_sentences = self.bucket_sentences(sentences, bucket_max_size=bucket_max_size)
        bucket_count = len(all_sentences)
        if verbose:
            print(">> sentences bucket_count:", bucket_count,
                  "bucket sizes:", [(len(s), [t["idx"] for t in s]) for s in all_sentences],
                  "bucket_max_size:", bucket_max_size)
        for sentences in all_sentences:
            temp_tokens: List[torch.Tensor] = []
            all_text_tokens.append(temp_tokens)
            for item in sentences:
                sent_tokens = item["sent"]
                # Clean up punctuation for TTS right before converting to IDs
                sent_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(sent_tokens))
                cleaned_sent_text = remove_tts_punctuation(sent_text)
                text_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(cleaned_sent_text))

                text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
                if verbose:
                    print(text_tokens)
                    print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                    # debug tokenizer
                    text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                    print("text_token_syms is same as sentence tokens", text_token_syms == sent_tokens) 
                temp_tokens.append(text_tokens)
        
            
        # Sequential processing of bucketing data
        all_batch_num = sum(len(s) for s in all_sentences)
        all_batch_codes = []
        processed_num = 0
        for item_tokens in all_text_tokens:
            batch_num = len(item_tokens)
            if batch_num > 1:
                batch_text_tokens = self.pad_tokens_cat(item_tokens)
            else:
                batch_text_tokens = item_tokens[0]
            processed_num += batch_num
            # gpt speech
            self._set_gr_progress(0.2 + 0.3 * processed_num/all_batch_num, f"gpt inference speech... {processed_num}/{all_batch_num}")
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(batch_text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    temp_codes = self.gpt.inference_speech(auto_conditioning, batch_text_tokens,
                                        cond_mel_lengths=cond_mel_lengths,
                                        # text_lengths=text_len,
                                        do_sample=do_sample,
                                        top_p=top_p,
                                        top_k=top_k,
                                        temperature=temperature,
                                        num_return_sequences=autoregressive_batch_size,
                                        length_penalty=length_penalty,
                                        num_beams=num_beams,
                                        repetition_penalty=repetition_penalty,
                                        max_generate_length=max_mel_tokens,
                                        **generation_kwargs)
                    all_batch_codes.append(temp_codes)
            gpt_gen_time += time.perf_counter() - m_start_time

        # gpt latent
        self._set_gr_progress(0.5, "gpt inference latents...")
        all_idxs = []
        all_latents = []
        has_warned = False
        for batch_codes, batch_tokens, batch_sentences in zip(all_batch_codes, all_text_tokens, all_sentences):
            for i in range(batch_codes.shape[0]):
                codes = batch_codes[i]  # [x]
                if not has_warned and codes[-1] != self.stop_mel_token:
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True
                codes = codes.unsqueeze(0)  # [x] -> [1, x]
                if verbose:
                    print("codes:", codes.shape)
                    print(codes)
                # 添加重试机制处理早期截断
                retry_count = 0
                max_retries = 3  # 使用3次重试
                all_attempts = []  # 保存所有尝试的结果
                
                while retry_count <= max_retries:
                    try:
                        # 计算当前codes的截断比例
                        if (codes == self.stop_mel_token).any():
                            stop_mel_idx = (codes == self.stop_mel_token).nonzero(as_tuple=False)
                            first_stop_position = stop_mel_idx[0, -1].item() if len(stop_mel_idx) > 0 else codes.size(-1)
                            current_ratio = first_stop_position / codes.size(-1)
                        else:
                            current_ratio = 1.0
                            first_stop_position = codes.size(-1)
                        
                        # 保存当前尝试
                        all_attempts.append({
                            'codes': codes.clone(),
                            'ratio': current_ratio,
                            'retry': retry_count
                        })
                        
                        # 打印当前尝试的截断占比
                        position_info = f"{first_stop_position}" if current_ratio < 1.0 else "full"
                        if 'i' in locals():
                            print(f"[ATTEMPT {retry_count}] Sentence {i}: Truncation ratio {current_ratio*100:.1f}% (position {position_info}/{codes.size(-1)})")
                        else:
                            print(f"[ATTEMPT {retry_count}] Truncation ratio {current_ratio*100:.1f}% (position {position_info}/{codes.size(-1)})")
                        
                        # 添加调试日志
                        codes_len = codes.shape[-1] if codes.dim() > 1 else codes.shape[0]
                        print(f"[DEBUG] codes shape: {codes.shape}, codes_len={codes_len}, max_mel_tokens={max_mel_tokens}")
                        print(f"[DEBUG] codes[-1]={codes.flatten()[-1].item() if codes_len > 0 else 'empty'}, stop_mel_token={self.stop_mel_token}")
                        print(f"[DEBUG] Has stop token: {(codes == self.stop_mel_token).any().item()}")
                        
                        # 检查异常情况
                        should_retry = False
                        retry_reason = ""
                        
                        # 1. 检查是否达到max_mel_tokens
                        if codes_len >= max_mel_tokens:
                            should_retry = True
                            retry_reason = f"Hit max_mel_tokens ({max_mel_tokens})"
                        
                        # 2. 检查静音占比是否过高
                        elif silence_ratio > 0.5:  # 超过50%是静音
                            should_retry = True
                            retry_reason = f"High silence ratio: {silence_ratio*100:.1f}%"
                        
                        # 3. 检查末尾是否有超长静音段
                        elif silence_segments and silence_segments[-1][1] == len(codes_flat):
                            last_silence_ratio = silence_segments[-1][2] / codes_len
                            if last_silence_ratio > 0.3:  # 末尾静音超过30%
                                should_retry = True
                                retry_reason = f"Large silence at end: {last_silence_ratio*100:.1f}%"
                        
                        if should_retry and retry_count < max_retries:
                            print(f"[WARNING] {retry_reason} - likely incomplete generation")
                            raise RuntimeError(f"{retry_reason}. This likely indicates incomplete generation.")
                        
                        # 统计静音token的数量和连续静音段
                        silence_count = (codes == 52).sum().item()
                        silence_ratio = silence_count / codes_len if codes_len > 0 else 0
                        print(f"[DEBUG] Silence tokens: {silence_count}/{codes_len} ({silence_ratio*100:.1f}%)")
                        
                        # 分析连续静音段
                        codes_flat = codes.flatten()
                        silence_segments = []
                        current_silence_start = None
                        
                        for i in range(len(codes_flat)):
                            if codes_flat[i] == 52:  # 静音token
                                if current_silence_start is None:
                                    current_silence_start = i
                            else:
                                if current_silence_start is not None:
                                    silence_length = i - current_silence_start
                                    if silence_length >= 10:  # 只记录超过10个token的静音段
                                        silence_segments.append((current_silence_start, i, silence_length))
                                    current_silence_start = None
                        
                        # 处理结尾的静音段
                        if current_silence_start is not None:
                            silence_length = len(codes_flat) - current_silence_start
                            if silence_length >= 10:
                                silence_segments.append((current_silence_start, len(codes_flat), silence_length))
                        
                        if silence_segments:
                            print(f"[DEBUG] Found {len(silence_segments)} long silence segments:")
                            for start, end, length in silence_segments[:3]:  # 只显示前3个
                                print(f"  - Position {start}-{end}: {length} tokens ({length/codes_len*100:.1f}% of total)")
                            
                            # 检查末尾是否有超长静音段
                            if silence_segments and silence_segments[-1][1] == len(codes_flat):
                                last_silence_ratio = silence_segments[-1][2] / codes_len
                                if last_silence_ratio > 0.3:  # 如果末尾静音超过30%
                                    print(f"[WARNING] Large silence segment at end: {last_silence_ratio*100:.1f}% of total length")
                        
                        # 分析token分布
                        unique_tokens, counts = torch.unique(codes, return_counts=True)
                        top_5_tokens = []
                        if len(unique_tokens) > 0:
                            sorted_indices = torch.argsort(counts, descending=True)[:5]
                            for idx in sorted_indices:
                                token = unique_tokens[idx].item()
                                count = counts[idx].item()
                                top_5_tokens.append(f"token_{token}:{count}")
                        print(f"[DEBUG] Top 5 tokens: {', '.join(top_5_tokens)}")
                        
                        codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                        print(f"[DEBUG] After remove_long_silence: code_lens={code_lens}")
                        break  # 成功（ratio >= 0.9），退出重试循环
                        
                    except RuntimeError as e:
                        if ("Early stop_mel_token" in str(e) or "hit max_mel_tokens" in str(e)) and retry_count < max_retries:
                            retry_count += 1
                            print(f"[RETRY {retry_count}/{max_retries}] Sentence {i}: {e}")
                            print(f"[RETRY {retry_count}/{max_retries}] Regenerating with adjusted parameters...")
                            
                            # 调整参数重新生成
                            adjusted_temperature = max(0.3, temperature * (0.8 - retry_count * 0.1))
                            adjusted_repetition_penalty = min(30.0, repetition_penalty * (1.3 + retry_count * 0.2))
                            
                            # 重新生成这个句子的codes
                            current_text_tokens = temp_tokens[i]
                            with torch.no_grad():
                                with torch.amp.autocast(current_text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                                    temp_codes = self.gpt.inference_speech(auto_conditioning, current_text_tokens,
                                                                    cond_mel_lengths=cond_mel_lengths,
                                                                    do_sample=do_sample,
                                                                    top_p=top_p,
                                                                    top_k=top_k,
                                                                    temperature=adjusted_temperature,
                                                                    num_return_sequences=autoregressive_batch_size,
                                                                    length_penalty=length_penalty,
                                                                    num_beams=num_beams,
                                                                    repetition_penalty=adjusted_repetition_penalty,
                                                                    max_generate_length=max_mel_tokens,
                                                                    **generation_kwargs)
                                    codes = temp_codes[i].unsqueeze(0)  # 提取对应的句子
                        else:
                            # 已达最大重试次数，从所有尝试中选择最佳结果
                            if all_attempts:
                                # 选择截断比例最高的结果
                                best_attempt = max(all_attempts, key=lambda x: x['ratio'])
                                
                                # 检查最佳结果是否达到70%的阈值
                                if best_attempt['ratio'] < 0.7:
                                    attempts_info = [(a['retry'], f"{a['ratio']*100:.1f}%") for a in all_attempts]
                                    print(f"[ERROR] Sentence {i}: All {len(all_attempts)} attempts failed to reach 70% completion threshold")
                                    print(f"[ERROR] All attempts: {attempts_info}")
                                    print(f"[ERROR] Best attempt only reached {best_attempt['ratio']*100:.1f}% completion")
                                    raise RuntimeError(f"Failed to generate complete speech after {len(all_attempts)} attempts. Best completion ratio: {best_attempt['ratio']*100:.1f}%")
                                
                                codes = best_attempt['codes']
                                print(f"[BEST RESULT] Sentence {i}: Selected attempt {best_attempt['retry']} with {best_attempt['ratio']*100:.1f}% completion")
                                attempts_info = [(a['retry'], f"{a['ratio']*100:.1f}%") for a in all_attempts]
                                print(f"[BEST RESULT] All attempts: {attempts_info}")
                                
                                # 使用最佳结果的实际长度
                                if (codes == self.stop_mel_token).any():
                                    stop_idx = (codes == self.stop_mel_token).nonzero(as_tuple=False)
                                    code_lens = torch.tensor([stop_idx[0, -1].item()], device=codes.device, dtype=torch.long)
                                else:
                                    code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=torch.long)
                            else:
                                # 保险起见，如果没有任何尝试记录
                                print(f"[ERROR] Sentence {i}: {e}")
                                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=torch.long)
                            break
                if verbose:
                    print("fix codes:", codes.shape)
                    print(codes)
                    print("code_lens:", code_lens)
                text_tokens = batch_tokens[i]
                all_idxs.append(batch_sentences[i]["idx"])
                m_start_time = time.perf_counter()
                with torch.no_grad():
                    with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                        latent = \
                            self.gpt(auto_conditioning, text_tokens,
                                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                        code_lens*self.gpt.mel_length_compression,
                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                        return_latent=True, clip_inputs=False)
                        gpt_forward_time += time.perf_counter() - m_start_time
                        all_latents.append(latent)
        del all_batch_codes, all_text_tokens, all_sentences
        # bigvgan chunk
        chunk_size = 2
        all_latents = [all_latents[all_idxs.index(i)] for i in range(len(all_latents))]
        if verbose:
            print(">> all_latents:", len(all_latents))
            print("  latents length:", [l.shape[1] for l in all_latents])
        chunk_latents = [all_latents[i : i + chunk_size] for i in range(0, len(all_latents), chunk_size)]
        chunk_length = len(chunk_latents)
        latent_length = len(all_latents)

        # bigvgan chunk decode
        self._set_gr_progress(0.7, "bigvgan decode...")
        tqdm_progress = tqdm(total=latent_length, desc="bigvgan")
        for items in chunk_latents:
            tqdm_progress.update(len(items))
            latent = torch.cat(items, dim=1)
            with torch.no_grad():
                with torch.amp.autocast(latent.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)
                    pass
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wavs.append(wav.cpu()) # to cpu before saving

        # clear cache
        tqdm_progress.close()  # 确保进度条被关闭
        del all_latents, chunk_latents
        end_time = time.perf_counter()
        self.torch_empty_cache()

        # wav audio output
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total fast inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> [fast] bigvgan chunk_length: {chunk_length}")
        print(f">> [fast] batch_num: {all_batch_num} bucket_max_size: {bucket_max_size}", f"bucket_count: {bucket_count}" if bucket_max_size > 1 else "")
        print(f">> [fast] RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            
            # --- Begin SRT Generation ---
            try:
                srt_path = os.path.splitext(output_path)[0] + ".srt"
                # In fast mode, sentences are bucketed, so we need to flatten them back
                
                # Reorder original_sentences to match the order of latents
                ordered_original_sentences = [None] * len(all_idxs)
                
                # Use all_idxs to reorder original_sentences
                for i, idx in enumerate(all_idxs):
                    if idx < len(original_sentences):
                        ordered_original_sentences[i] = original_sentences[idx]

                self.generate_srt(srt_path, ordered_original_sentences, all_latents, sampling_rate)
                print(">> srt file saved to:", srt_path)
            except Exception as e:
                print(f">> Failed to generate SRT file: {e}")
            # --- End SRT Generation ---

            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)

    def generate_srt(self, srt_path, sentences, all_latents, sampling_rate):
        def format_time(seconds):
            """Converts seconds to SRT time format HH:MM:SS,ms"""
            hours, remainder = divmod(seconds, 3600)
            minutes, remainder = divmod(remainder, 60)
            seconds, milliseconds = divmod(remainder, 1)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds*1000):03}"

        total_duration_s = 0
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            for i, (sentence, latent) in enumerate(zip(sentences, all_latents), 1):
                # The duration of a latent is its length * mel_length_compression / sampling_rate
                duration_s = latent.shape[1] * self.gpt.mel_length_compression / sampling_rate
                
                start_time = total_duration_s
                end_time = total_duration_s + duration_s
                
                # sentence is already a string (original text)
                sentence_text = sentence

                srt_file.write(f"{i}\n")
                srt_file.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
                srt_file.write(f"{sentence_text.strip()}\n\n")
                
                total_duration_s = end_time


    # 原始推理模式
    def infer(self, audio_prompt, text, output_path, verbose=False, max_text_tokens_per_sentence=200, **generation_kwargs):
        print(">> start inference...")
        self._set_gr_progress(0, "start inference...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass

        self._set_gr_progress(0.1, "text processing...")
        auto_conditioning = cond_mel
        
        # Store original text for SRT generation
        original_text = text
        
        # For TTS processing, use normalized tokens
        normalized_text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(normalized_text_tokens_list, max_text_tokens_per_sentence)
        
        # Extract original sentences based on character positions
        # This approach preserves all original punctuation and formatting
        original_sentences = self._extract_original_sentences(original_text, sentences)

        if verbose:
            print("text token count:", len(normalized_text_tokens_list))
            print("sentences count:", len(sentences))
            print("max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        # 动态计算max_mel_tokens，但不能超过模型训练时的限制
        # 模型配置中 max_mel_tokens=605, max_text_tokens=402
        model_max_mel_tokens = self.cfg.gpt.max_mel_tokens  # 605
        default_max_mel = min(model_max_mel_tokens - 5, max(600, max_text_tokens_per_sentence * 3))  # 留5个token余量，避免越界
        
        # 调试信息
        print(f"[DEBUG] model_max_mel_tokens from config: {model_max_mel_tokens}")
        print(f"[DEBUG] max_text_tokens_per_sentence: {max_text_tokens_per_sentence}")
        print(f"[DEBUG] calculated default_max_mel: {default_max_mel}")
        print(f"[DEBUG] generation_kwargs before pop: {generation_kwargs}")
        
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", default_max_mel)
        
        print(f"[DEBUG] max_mel_tokens after pop: {max_mel_tokens}")
        
        # 确保不超过模型限制
        if max_mel_tokens > model_max_mel_tokens - 5:
            print(f"[WARNING] Requested max_mel_tokens ({max_mel_tokens}) exceeds model limit ({model_max_mel_tokens}), capping to {model_max_mel_tokens - 5}")
            max_mel_tokens = model_max_mel_tokens - 5
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        all_latents = [] # <--- Add this line
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0
        progress = 0
        has_warned = False
        for sent in sentences:
            # Clean up punctuation for TTS right before converting to IDs
            sent_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(sent))
            cleaned_sent_text = remove_tts_punctuation(sent_text)
            text_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(cleaned_sent_text))

            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            # text_tokens = F.pad(text_tokens, (1, 0), value=0)
            # text_tokens = F.pad(text_tokens, (0, 1), value=1)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as sentence tokens", text_token_syms == sent)

            # text_len = torch.IntTensor([text_tokens.size(1)], device=text_tokens.device)
            # print(text_len)
            progress += 1
            self._set_gr_progress(0.2 + 0.4 * (progress-1) / len(sentences), f"gpt inference latent... {progress}/{len(sentences)}")
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                      device=text_tokens.device),
                                                        # text_lengths=text_len,
                                                        do_sample=do_sample,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        temperature=temperature,
                                                        num_return_sequences=autoregressive_batch_size,
                                                        length_penalty=length_penalty,
                                                        num_beams=num_beams,
                                                        repetition_penalty=repetition_penalty,
                                                        max_generate_length=max_mel_tokens,
                                                        **generation_kwargs)
                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                if verbose:
                    print(codes, type(codes))
                    print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                # remove ultra-long silence if exits
                # temporarily fix the long silence bug.
                # 添加重试机制处理早期截断
                max_retries = 3
                retry_count = 0
                all_attempts = []  # 保存所有尝试的结果
                
                while retry_count <= max_retries:
                    try:
                        # 计算当前codes的截断比例
                        if (codes == self.stop_mel_token).any():
                            stop_mel_idx = (codes == self.stop_mel_token).nonzero(as_tuple=False)
                            first_stop_position = stop_mel_idx[0, -1].item() if len(stop_mel_idx) > 0 else codes.size(-1)
                            current_ratio = first_stop_position / codes.size(-1)
                        else:
                            current_ratio = 1.0
                            first_stop_position = codes.size(-1)
                        
                        # 保存当前尝试
                        all_attempts.append({
                            'codes': codes.clone(),
                            'ratio': current_ratio,
                            'retry': retry_count
                        })
                        
                        # 打印当前尝试的截断占比
                        position_info = f"{first_stop_position}" if current_ratio < 1.0 else "full"
                        if 'i' in locals():
                            print(f"[ATTEMPT {retry_count}] Sentence {i}: Truncation ratio {current_ratio*100:.1f}% (position {position_info}/{codes.size(-1)})")
                        else:
                            print(f"[ATTEMPT {retry_count}] Truncation ratio {current_ratio*100:.1f}% (position {position_info}/{codes.size(-1)})")
                        
                        # 添加调试日志
                        codes_len = codes.shape[-1] if codes.dim() > 1 else codes.shape[0]
                        print(f"[DEBUG] codes shape: {codes.shape}, codes_len={codes_len}, max_mel_tokens={max_mel_tokens}")
                        print(f"[DEBUG] codes[-1]={codes.flatten()[-1].item() if codes_len > 0 else 'empty'}, stop_mel_token={self.stop_mel_token}")
                        print(f"[DEBUG] Has stop token: {(codes == self.stop_mel_token).any().item()}")
                        
                        # 检查异常情况
                        should_retry = False
                        retry_reason = ""
                        
                        # 1. 检查是否达到max_mel_tokens
                        if codes_len >= max_mel_tokens:
                            should_retry = True
                            retry_reason = f"Hit max_mel_tokens ({max_mel_tokens})"
                        
                        # 2. 检查静音占比是否过高
                        elif silence_ratio > 0.5:  # 超过50%是静音
                            should_retry = True
                            retry_reason = f"High silence ratio: {silence_ratio*100:.1f}%"
                        
                        # 3. 检查末尾是否有超长静音段
                        elif silence_segments and silence_segments[-1][1] == len(codes_flat):
                            last_silence_ratio = silence_segments[-1][2] / codes_len
                            if last_silence_ratio > 0.3:  # 末尾静音超过30%
                                should_retry = True
                                retry_reason = f"Large silence at end: {last_silence_ratio*100:.1f}%"
                        
                        if should_retry and retry_count < max_retries:
                            print(f"[WARNING] {retry_reason} - likely incomplete generation")
                            raise RuntimeError(f"{retry_reason}. This likely indicates incomplete generation.")
                        
                        # 统计静音token的数量和连续静音段
                        silence_count = (codes == 52).sum().item()
                        silence_ratio = silence_count / codes_len if codes_len > 0 else 0
                        print(f"[DEBUG] Silence tokens: {silence_count}/{codes_len} ({silence_ratio*100:.1f}%)")
                        
                        # 分析连续静音段
                        codes_flat = codes.flatten()
                        silence_segments = []
                        current_silence_start = None
                        
                        for i in range(len(codes_flat)):
                            if codes_flat[i] == 52:  # 静音token
                                if current_silence_start is None:
                                    current_silence_start = i
                            else:
                                if current_silence_start is not None:
                                    silence_length = i - current_silence_start
                                    if silence_length >= 10:  # 只记录超过10个token的静音段
                                        silence_segments.append((current_silence_start, i, silence_length))
                                    current_silence_start = None
                        
                        # 处理结尾的静音段
                        if current_silence_start is not None:
                            silence_length = len(codes_flat) - current_silence_start
                            if silence_length >= 10:
                                silence_segments.append((current_silence_start, len(codes_flat), silence_length))
                        
                        if silence_segments:
                            print(f"[DEBUG] Found {len(silence_segments)} long silence segments:")
                            for start, end, length in silence_segments[:3]:  # 只显示前3个
                                print(f"  - Position {start}-{end}: {length} tokens ({length/codes_len*100:.1f}% of total)")
                            
                            # 检查末尾是否有超长静音段
                            if silence_segments and silence_segments[-1][1] == len(codes_flat):
                                last_silence_ratio = silence_segments[-1][2] / codes_len
                                if last_silence_ratio > 0.3:  # 如果末尾静音超过30%
                                    print(f"[WARNING] Large silence segment at end: {last_silence_ratio*100:.1f}% of total length")
                        
                        # 分析token分布
                        unique_tokens, counts = torch.unique(codes, return_counts=True)
                        top_5_tokens = []
                        if len(unique_tokens) > 0:
                            sorted_indices = torch.argsort(counts, descending=True)[:5]
                            for idx in sorted_indices:
                                token = unique_tokens[idx].item()
                                count = counts[idx].item()
                                top_5_tokens.append(f"token_{token}:{count}")
                        print(f"[DEBUG] Top 5 tokens: {', '.join(top_5_tokens)}")
                        
                        codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                        print(f"[DEBUG] After remove_long_silence: code_lens={code_lens}")
                        break  # 成功（ratio >= 0.9），退出重试循环
                        
                    except RuntimeError as e:
                        if ("Early stop_mel_token" in str(e) or "hit max_mel_tokens" in str(e)) and retry_count < max_retries:
                            retry_count += 1
                            print(f"[RETRY {retry_count}/{max_retries}] {e}")
                            print(f"[RETRY {retry_count}/{max_retries}] Regenerating with adjusted parameters...")
                            
                            # 调整参数重新生成
                            adjusted_temperature = max(0.3, temperature * (0.8 - retry_count * 0.1))
                            adjusted_repetition_penalty = min(30.0, repetition_penalty * (1.3 + retry_count * 0.2))
                            
                            # 重新生成codes
                            with torch.no_grad():
                                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                                    codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                                      device=text_tokens.device),
                                                                        do_sample=do_sample,
                                                                        top_p=top_p,
                                                                        top_k=top_k,
                                                                        temperature=adjusted_temperature,
                                                                        num_return_sequences=autoregressive_batch_size,
                                                                        length_penalty=length_penalty,
                                                                        num_beams=num_beams,
                                                                        repetition_penalty=adjusted_repetition_penalty,
                                                                        max_generate_length=max_mel_tokens,
                                                                        **generation_kwargs)
                        else:
                            # 已达最大重试次数，从所有尝试中选择最佳结果
                            if all_attempts:
                                # 选择截断比例最高的结果
                                best_attempt = max(all_attempts, key=lambda x: x['ratio'])
                                
                                # 检查最佳结果是否达到70%的阈值
                                if best_attempt['ratio'] < 0.7:
                                    attempts_info = [(a['retry'], f"{a['ratio']*100:.1f}%") for a in all_attempts]
                                    print(f"[ERROR] All {len(all_attempts)} attempts failed to reach 70% completion threshold")
                                    print(f"[ERROR] All attempts: {attempts_info}")
                                    print(f"[ERROR] Best attempt only reached {best_attempt['ratio']*100:.1f}% completion")
                                    raise RuntimeError(f"Failed to generate complete speech after {len(all_attempts)} attempts. Best completion ratio: {best_attempt['ratio']*100:.1f}%")
                                
                                codes = best_attempt['codes']
                                print(f"[BEST RESULT] Selected attempt {best_attempt['retry']} with {best_attempt['ratio']*100:.1f}% completion")
                                attempts_info = [(a['retry'], f"{a['ratio']*100:.1f}%") for a in all_attempts]
                                print(f"[BEST RESULT] All attempts: {attempts_info}")
                                
                                # 使用最佳结果的实际长度
                                if (codes == self.stop_mel_token).any():
                                    stop_idx = (codes == self.stop_mel_token).nonzero(as_tuple=False)
                                    code_lens = torch.tensor([stop_idx[0, -1].item()], device=codes.device, dtype=torch.long)
                                else:
                                    code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=torch.long)
                            else:
                                # 保险起见，如果没有任何尝试记录
                                print(f"[ERROR] {e}")
                                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=torch.long)
                            break
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")
                self._set_gr_progress(0.2 + 0.4 * progress / len(sentences), f"gpt inference speech... {progress}/{len(sentences)}")
                m_start_time = time.perf_counter()
                # latent, text_lens_out, code_lens_out = \
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = \
                        self.gpt(auto_conditioning, text_tokens,
                                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                    code_lens*self.gpt.mel_length_compression,
                                    cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                    return_latent=True, clip_inputs=False)
                    all_latents.append(latent) # <--- Add this line
                    gpt_forward_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
        end_time = time.perf_counter()
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)

            # --- Begin SRT Generation ---
            try:
                srt_path = os.path.splitext(output_path)[0] + ".srt"
                self.generate_srt(srt_path, original_sentences, all_latents, sampling_rate)
                print(">> srt file saved to:", srt_path)
            except Exception as e:
                print(f">> Failed to generate SRT file: {e}")
            # --- End SRT Generation ---

            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)


if __name__ == "__main__":
    prompt_wav="test_data/input.wav"
    #text="晕 XUAN4 是 一 种 GAN3 觉"
    #text='大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！'
    text="There is a vehicle arriving in dock number 7?"

    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, use_cuda_kernel=False)
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
