#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to verify that original text is preserved in SRT files"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indextts.utils.front import TextNormalizer, TextTokenizer

def test_tokenizer():
    print("=== Testing Tokenizer ===")
    
    # Initialize
    text_normalizer = TextNormalizer()
    text_normalizer.load()
    tokenizer = TextTokenizer(
        vocab_file="checkpoints/bpe.model",
        normalizer=text_normalizer,
    )
    
    # Test cases with Chinese punctuation and special characters
    test_texts = [
        "你好！这是一个测试。",
        "这是「引号」和《书名号》的测试。",
        "这里有：冒号；分号、顿号……省略号",
        "Testing English! And Chinese？混合测试。",
    ]
    
    for text in test_texts:
        print(f"\nOriginal text: {text}")
        
        # Tokenize without normalization (for SRT)
        original_tokens = tokenizer.tokenize_without_normalize(text)
        original_sentences = tokenizer.split_sentences_original(original_tokens, max_tokens_per_sentence=50)
        
        # Tokenize with normalization (for TTS)
        normalized_tokens = tokenizer.tokenize(text)
        normalized_sentences = tokenizer.split_sentences(normalized_tokens, max_tokens_per_sentence=50)
        
        print(f"Original tokens: {original_tokens}")
        print(f"Original sentences: {original_sentences}")
        print(f"Normalized tokens: {normalized_tokens}")
        print(f"Normalized sentences: {normalized_sentences}")
        
        # Decode back to verify
        for i, (orig_sent, norm_sent) in enumerate(zip(original_sentences, normalized_sentences)):
            orig_ids = tokenizer.convert_tokens_to_ids(orig_sent)
            norm_ids = tokenizer.convert_tokens_to_ids(norm_sent)
            
            orig_decoded = tokenizer.decode(orig_ids)
            norm_decoded = tokenizer.decode(norm_ids)
            
            print(f"\n  Sentence {i+1}:")
            print(f"    Original decoded: {orig_decoded}")
            print(f"    Normalized decoded: {norm_decoded}")

def test_punctuation_marks():
    print("\n\n=== Testing Punctuation Marks ===")
    
    text_normalizer = TextNormalizer()
    text_normalizer.load()
    tokenizer = TextTokenizer(
        vocab_file="checkpoints/bpe.model", 
        normalizer=text_normalizer,
    )
    
    print("Original punctuation marks tokens:")
    print(tokenizer.original_punctuation_marks_tokens)
    
    print("\nNormalized punctuation marks tokens:")
    print(tokenizer.punctuation_marks_tokens)
    
    # Check if Chinese punctuation can be tokenized
    chinese_punctuation = ["。", "！", "？", "，", "：", "；"]
    print("\nTesting Chinese punctuation tokenization:")
    for punct in chinese_punctuation:
        tokens = tokenizer.sp_model.Encode(punct, out_type=str)
        ids = tokenizer.sp_model.Encode(punct, out_type=int)
        decoded = tokenizer.sp_model.Decode(ids)
        print(f"  {punct} -> tokens: {tokens}, ids: {ids}, decoded: '{decoded}'")

if __name__ == "__main__":
    test_tokenizer()
    test_punctuation_marks()
    print("\n=== Tests completed ===")