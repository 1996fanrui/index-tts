#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the full pipeline to ensure original text is preserved in SRT"""

import os
import sys
import tempfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indextts.infer import IndexTTS

def test_srt_generation():
    print("=== Testing SRT Generation with Original Text ===")
    
    # Test texts with various punctuation marks
    test_cases = [
        "你好！这是一个测试。包含各种标点符号。",
        "这是「引号」和《书名号》，还有：冒号、顿号；分号……省略号。",
        "第一句话。第二句话！第三句话？",
    ]
    
    # Initialize TTS
    tts = IndexTTS(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        is_fp16=True,
        device="cpu"  # Use CPU for testing
    )
    
    # Use a test audio prompt
    prompt_wav = "test_data/input.wav"
    
    for i, text in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input text: {text}")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            output_path = tmp_wav.name
        
        try:
            # Run inference
            tts.infer(
                audio_prompt=prompt_wav,
                text=text,
                output_path=output_path,
                verbose=False,
                max_text_tokens_per_sentence=50
            )
            
            # Check if SRT file was created
            srt_path = os.path.splitext(output_path)[0] + ".srt"
            if os.path.exists(srt_path):
                print(f"\nSRT file created: {srt_path}")
                with open(srt_path, 'r', encoding='utf-8') as f:
                    srt_content = f.read()
                print("SRT content:")
                print(srt_content)
                
                # Verify original punctuation is preserved
                original_chars = ["。", "！", "？", "「", "」", "《", "》", "：", "、", "；", "……"]
                preserved_chars = []
                missing_chars = []
                
                for char in original_chars:
                    if char in text:
                        if char in srt_content:
                            preserved_chars.append(char)
                        else:
                            missing_chars.append(char)
                
                if preserved_chars:
                    print(f"\nPreserved characters: {', '.join(preserved_chars)}")
                if missing_chars:
                    print(f"Missing characters: {', '.join(missing_chars)}")
                
                # Cleanup
                os.remove(srt_path)
            else:
                print("ERROR: SRT file was not created")
                
        except Exception as e:
            print(f"ERROR during inference: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)

if __name__ == "__main__":
    # Check if required files exist
    if not os.path.exists("checkpoints/config.yaml"):
        print("ERROR: checkpoints/config.yaml not found")
        print("Please ensure you have the model checkpoints in the correct location")
        sys.exit(1)
    
    if not os.path.exists("test_data/input.wav"):
        print("ERROR: test_data/input.wav not found")
        print("Please ensure you have a test audio file")
        sys.exit(1)
    
    test_srt_generation()
    print("\n=== Testing completed ===")