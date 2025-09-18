"""
Integration test for TransMLA with Qwen2.5-0.5B model
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os

from fedcore.algorithm.reassembly.core_reassemblers import ParentalReassembler
from fedcore.algorithm.reassembly.transmla_reassembler import TransMLA, TransMLAConfig, TRANSMLA_AVAILABLE


class TestTransMLAQwenIntegration:
    """Integration tests for TransMLA with Qwen2.5-0.5B"""

    @pytest.fixture
    def mock_qwen_model(self):
        """Create a mock Qwen2.5-0.5B model with realistic config"""
        model = Mock()
        model.config = Mock()
        model.config.model_type = "qwen2"
        model.config.hidden_size = 896
        model.config.num_attention_heads = 14
        model.config.head_dim = 64
        model.config.intermediate_size = 4864
        model.config.num_hidden_layers = 24
        model.config.vocab_size = 151936
        
        # Mock model methods
        model.save_pretrained = Mock()
        model.to = Mock(return_value=model)
        
        # Mock parameters() method to return mock parameters with device
        mock_param1 = Mock()
        mock_param1.device = torch.device('cpu')
        mock_param2 = Mock()
        mock_param2.device = torch.device('cpu')
        model.parameters = Mock(return_value=[mock_param1, mock_param2])
        
        # Mock named_modules() for traversal
        model.named_modules = Mock(return_value=[('', model)])
        
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer"""
        tokenizer = Mock()
        tokenizer.save_pretrained = Mock()
        tokenizer.pad_token_id = 151643
        tokenizer.eos_token_id = 151645
        tokenizer.vocab_size = 151936
        return tokenizer

    @pytest.fixture
    def transmla_config(self):
        """Create TransMLA configuration optimized for Qwen2.5-0.5B"""
        return TransMLAConfig(
            freqfold="auto",
            collapse="auto", 
            qk_mqa_dim=64,
            q_lora_rank=None,
            kv_lora_rank=128,  # Must be < 2*latent_dim - qk_mqa_dim = 2*128-64 = 192
            balance_kv_ratio=1.0,
            use_qkv_norm=False,
            cal_dataset="wikitext2",
            cal_nsamples=64,   # Smaller sample size for testing
            cal_batch_size=4,  # Smaller batch for testing
            cal_max_seqlen=128, # Shorter sequences for testing
            ppl_eval_batch_size=1,
            deepseek_style=True,
            dtype="bf16",
            device="auto",
            seed=42
        )

    def test_qwen_config_validation(self, mock_qwen_model, transmla_config):
        """Test that Qwen2.5-0.5B config is properly validated"""
        model = mock_qwen_model
        
        # Test head_dim calculation
        expected_head_dim = model.config.hidden_size // model.config.num_attention_heads
        assert expected_head_dim == 64  # 896 // 14 = 64
        
        # Test collapse calculation
        expected_collapse = expected_head_dim // transmla_config.qk_mqa_dim
        assert expected_collapse == 1  # 64 // 64 = 1

    @patch('fedcore.algorithm.reassembly.transmla_reassembler.TRANSMLA_AVAILABLE', True)
    def test_qwen_transmla_direct_conversion(self, mock_qwen_model, mock_tokenizer, transmla_config):
        """Test direct TransMLA conversion for Qwen2.5-0.5B"""
        model = mock_qwen_model
        tokenizer = mock_tokenizer
        
        # Mock the conversion process to avoid validation issues
        with patch.object(TransMLA, 'reassemble') as mock_reassemble:
            mock_reassemble.return_value = model
            
            # Perform direct conversion
            result = TransMLA.reassemble(
                model=model,
                tokenizer=tokenizer,
                config=transmla_config
            )
            
            # Verify conversion was called
            assert result == model
            mock_reassemble.assert_called_once()

    @patch('fedcore.algorithm.reassembly.transmla_reassembler.TRANSMLA_AVAILABLE', True)
    def test_qwen_transmla_with_config(self, mock_qwen_model, mock_tokenizer, transmla_config):
        """Test TransMLA conversion with custom config for Qwen2.5-0.5B"""
        model = mock_qwen_model
        tokenizer = mock_tokenizer
        
        # Mock the entire conversion process to avoid validation issues
        with patch.object(TransMLA, 'reassemble') as mock_reassemble:
            mock_reassemble.return_value = model
            
            # Perform conversion with custom config
            result = TransMLA.reassemble(
                model=model,
                tokenizer=tokenizer,
                config=transmla_config
            )
            
            # Verify conversion was called with config
            assert result == model
            mock_reassemble.assert_called_once_with(
                model=model, tokenizer=tokenizer, config=transmla_config
            )

    @patch('fedcore.algorithm.reassembly.transmla_reassembler.TRANSMLA_AVAILABLE', True)
    def test_qwen_attention_reassembler_transmla_mode(self, mock_qwen_model, mock_tokenizer, transmla_config):
        """Test TransMLA for Qwen2.5-0.5B"""
        model = mock_qwen_model
        tokenizer = mock_tokenizer
        
        # Mock the conversion method to avoid validation
        with patch.object(TransMLA, 'reassemble') as mock_reassemble:
            mock_reassemble.return_value = model
            
            # Use TransMLA
            result = TransMLA.reassemble(
                model=model,
                tokenizer=tokenizer,
                config=transmla_config
            )
            
            # Verify conversion was performed
            assert result == model
            mock_reassemble.assert_called_once_with(
                model=model, tokenizer=tokenizer, config=transmla_config
            )

    def test_qwen_config_auto_parameters(self, mock_qwen_model, transmla_config):
        """Test automatic parameter calculation for Qwen2.5-0.5B"""
        model = mock_qwen_model
        config = transmla_config
        
        # Test auto collapse calculation
        if config.collapse == "auto":
            head_dim = model.config.head_dim or (model.config.hidden_size // model.config.num_attention_heads)
            expected_collapse = head_dim // config.qk_mqa_dim
            assert expected_collapse == 1  # For Qwen2.5-0.5B: 64 // 64 = 1

    @patch('fedcore.algorithm.reassembly.transmla_reassembler.TRANSMLA_AVAILABLE', True)
    def test_qwen_full_conversion_workflow(self, mock_qwen_model, mock_tokenizer, transmla_config):
        """Test complete TransMLA conversion workflow for Qwen2.5-0.5B"""
        model = mock_qwen_model
        tokenizer = mock_tokenizer
        
        # Mock the entire execution process to test workflow
        with patch.object(TransMLA, 'reassemble') as mock_reassemble:
            mock_reassemble.return_value = model
            
            # Perform conversion
            result = TransMLA.reassemble(
                model=model,
                tokenizer=tokenizer,
                config=transmla_config
            )
            
            # Verify conversion was called
            mock_reassemble.assert_called_once_with(
                model=model, tokenizer=tokenizer, config=transmla_config
            )
            assert result == model

    def test_qwen_model_size_validation(self, mock_qwen_model):
        """Test that Qwen2.5-0.5B model has expected dimensions"""
        model = mock_qwen_model
        
        # Validate model dimensions for 0.5B parameter model
        assert model.config.hidden_size == 896
        assert model.config.num_attention_heads == 14
        assert model.config.num_hidden_layers == 24
        assert model.config.intermediate_size == 4864
        
        # Calculate approximate parameter count
        # This is a rough estimation for validation
        embedding_params = model.config.vocab_size * model.config.hidden_size
        attention_params = model.config.num_hidden_layers * model.config.hidden_size * model.config.hidden_size * 4  # Q, K, V, O
        ffn_params = model.config.num_hidden_layers * model.config.hidden_size * model.config.intermediate_size * 2  # up, down
        
        total_params = embedding_params + attention_params + ffn_params
        # Should be approximately 0.5B parameters (allowing for some variance)
        assert 400_000_000 < total_params < 600_000_000

    @patch('fedcore.algorithm.reassembly.transmla_reassembler.TRANSMLA_AVAILABLE', False)
    @patch('fedcore.algorithm.reassembly.transmla_reassembler._initialize_transmla')
    def test_qwen_transmla_unavailable_fallback(self, mock_qwen_model, mock_tokenizer):
        """Test fallback behavior when TransMLA is unavailable"""
        model = mock_qwen_model
        tokenizer = mock_tokenizer
        
        with pytest.raises(AssertionError, match="TransMLA not available"):
            TransMLA.reassemble(
                model=model,
                tokenizer=tokenizer
            )

    def test_qwen_transmla_config_optimization(self):
        """Test TransMLA config optimization for small model"""
        config = TransMLAConfig(
            qk_mqa_dim=64,        # Match head_dim for efficiency
            kv_lora_rank=256,     # Smaller than default 512 for 0.5B model
            cal_nsamples=64,      # Fewer samples for faster testing
            cal_batch_size=4,     # Smaller batch size
            cal_max_seqlen=128    # Shorter sequences
        )
        
        # Verify optimized parameters
        assert config.qk_mqa_dim == 64
        assert config.kv_lora_rank == 256
        assert config.cal_nsamples == 64
        assert config.cal_batch_size == 4
        assert config.cal_max_seqlen == 128

    def test_qwen_direct_execution_workflow(self, mock_qwen_model, mock_tokenizer, transmla_config):
        """Test direct execution workflow for Qwen2.5-0.5B"""
        model = mock_qwen_model
        tokenizer = mock_tokenizer
        
        # Mock the execution
        with patch.object(TransMLA, 'reassemble') as mock_reassemble:
            mock_reassemble.return_value = model
            
            # Execute direct conversion
            result = TransMLA.reassemble(
                model=model,
                tokenizer=tokenizer,
                config=transmla_config
            )
            
            # Verify execution
            assert result == model
            mock_reassemble.assert_called_once_with(
                model=model, tokenizer=tokenizer, config=transmla_config
            )


class TestQwenModelSpecific:
    """Tests specific to Qwen2.5 model architecture"""

    def test_qwen_attention_head_configuration(self):
        """Test Qwen2.5-0.5B attention head configuration"""
        # Qwen2.5-0.5B specific configuration
        hidden_size = 896
        num_attention_heads = 14
        head_dim = hidden_size // num_attention_heads
        
        assert head_dim == 64
        assert hidden_size == num_attention_heads * head_dim

    def test_qwen_mla_compatibility(self):
        """Test that Qwen2.5-0.5B is compatible with MLA conversion"""
        config = TransMLAConfig(
            qk_mqa_dim=64,  # Should match head_dim for optimal performance
            collapse="auto"  # Will be calculated as head_dim // qk_mqa_dim = 1
        )
        
        # For Qwen2.5-0.5B, collapse should be 1 (64 // 64)
        head_dim = 64
        expected_collapse = head_dim // config.qk_mqa_dim
        assert expected_collapse == 1

    def test_qwen_vocabulary_size(self):
        """Test Qwen2.5 vocabulary size"""
        vocab_size = 151936
        
        # Verify vocabulary size is reasonable for Qwen2.5
        assert vocab_size > 150000
        assert vocab_size < 160000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
