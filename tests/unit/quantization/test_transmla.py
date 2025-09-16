"""
Tests for TransMLA and AttentionReassembler classes
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from fedcore.algorithm.reassembly.core_reassemblers import AttentionReassembler, get_reassembler, convert_model
from fedcore.algorithm.reassembly.transmla_reassembler import TransMLA, TransMLAConfig, TRANSMLA_AVAILABLE


class SimpleModel(nn.Module):
    """Simple test model"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.config = Mock()
        self.config.hidden_size = 768
        self.config.num_attention_heads = 12
        self.config.head_dim = 64

    def forward(self, x):
        return self.linear(x)


class TestAttentionReassembler:
    """Test AttentionReassembler functionality"""

    def test_convert_parental_mode(self):
        """Test parental conversion mode"""
        model = SimpleModel()
        
        # Should work without any special dependencies
        result = AttentionReassembler.convert(model, mode='parental')
        
        assert result is not None
        assert isinstance(result, nn.Module)

    def test_convert_unknown_mode(self):
        """Test that unknown mode raises key error"""
        model = SimpleModel()
        
        with pytest.raises(KeyError):
            AttentionReassembler.convert(model, mode='unknown_mode')

    @patch('fedcore.algorithm.reassembly.transmla_reassembler.TRANSMLA_AVAILABLE', True)
    def test_convert_transmla_mode_success(self):
        """Test TransMLA conversion mode when TransMLA is available"""
        model = SimpleModel()
        tokenizer = Mock()
        
        with patch('fedcore.algorithm.reassembly.transmla_reassembler.convert_model_to_mla') as mock_convert:
            mock_convert.return_value = model
            
            result = AttentionReassembler.convert(
                model, 
                mode='trans-mla', 
                tokenizer=tokenizer
            )
            
            assert result is not None
            mock_convert.assert_called_once()

    def test_convert_transmla_mode_no_tokenizer(self):
        """Test that TransMLA mode requires tokenizer"""
        model = SimpleModel()
        
        with pytest.raises(AssertionError, match="TransMLA conversion requires tokenizer"):
            AttentionReassembler.convert(model, mode='trans-mla')

    @patch('fedcore.algorithm.reassembly.transmla_reassembler.TRANSMLA_AVAILABLE', False)
    @patch('fedcore.algorithm.reassembly.transmla_reassembler._initialize_transmla')
    def test_convert_transmla_mode_unavailable(self, mock_init_transmla):
        """Test TransMLA mode when TransMLA is not available"""
        model = SimpleModel()
        tokenizer = Mock()

        with pytest.raises(AssertionError, match="TransMLA not available"):
            AttentionReassembler.convert(
                model,
                mode='trans-mla',
                tokenizer=tokenizer
            )


class TestTransMLA:
    """Test TransMLA functionality"""

    def test_convert_direct_execution(self):
        """Test direct execution"""
        model = SimpleModel()
        tokenizer = Mock()
        
        with patch.object(TransMLA, '_convert_trans_mla') as mock_convert:
            mock_convert.return_value = model
            
            result = TransMLA.convert(
                model, 
                tokenizer=tokenizer
            )
            
            assert result == model
            mock_convert.assert_called_once()

    @patch('fedcore.algorithm.reassembly.transmla_reassembler.TRANSMLA_AVAILABLE', True)
    def test_convert_with_mla_available(self):
        """Test successful conversion when TransMLA is available"""
        model = SimpleModel()
        tokenizer = Mock()
        
        with patch('fedcore.algorithm.reassembly.transmla_reassembler.convert_model_to_mla') as mock_convert:
            mock_convert.return_value = model
            
            result = TransMLA.convert(
                model=model,
                tokenizer=tokenizer
            )
            
            assert result == model




class TestReassemblerFunctions:
    """Test reassembler functions functionality"""

    def test_get_reassembler_valid_types(self):
        """Test getting valid reassembler types"""
        valid_types = ['attention', 'trans-mla', 'parental']
        
        for reassembler_type in valid_types:
            reassembler = get_reassembler(reassembler_type)
            assert reassembler is not None

    def test_get_reassembler_invalid_type(self):
        """Test getting invalid reassembler type raises ValueError"""
        with pytest.raises(ValueError, match="Unknown reassembler type"):
            get_reassembler('invalid_type')

    def test_convert_model_attention(self):
        """Test converting model with attention reassembler"""
        model = SimpleModel()
        
        with patch.object(AttentionReassembler, 'convert') as mock_convert:
            mock_convert.return_value = model
            
            result = convert_model(
                model, 
                'attention', 
                mode='parental'
            )
            
            assert result == model
            mock_convert.assert_called_once_with(model, mode='parental')

    def test_convert_model_transmla(self):
        """Test converting model with TransMLA reassembler"""
        model = SimpleModel()
        tokenizer = Mock()
        
        with patch.object(TransMLA, 'convert') as mock_convert:
            mock_convert.return_value = model
            
            result = convert_model(
                model, 
                'trans-mla',
                tokenizer=tokenizer
            )
            
            assert result == model
            mock_convert.assert_called_once_with(model, tokenizer=tokenizer)

    def test_convert_model_parental(self):
        """Test converting model with parental reassembler"""
        from fedcore.algorithm.reassembly.core_reassemblers import ParentalReassembler
        model = SimpleModel()
        
        with patch.object(ParentalReassembler, 'reassemble') as mock_reassemble:
            mock_reassemble.return_value = model
            
            result = convert_model(model, 'parental')
            
            assert result == model
            mock_reassemble.assert_called_once_with(model)


class TestTransMLAConfig:
    """Test TransMLAConfig functionality"""

    def test_default_config(self):
        """Test default configuration values"""
        config = TransMLAConfig()
        
        assert config.freqfold == "auto"
        assert config.collapse == "auto"
        assert config.qk_mqa_dim == 64
        assert config.kv_lora_rank == 512
        assert config.cal_dataset == "wikitext2"
        assert config.cal_nsamples == 128

    def test_custom_config(self):
        """Test custom configuration values"""
        config = TransMLAConfig(
            qk_mqa_dim=128,
            kv_lora_rank=256,
            cal_dataset="custom_dataset"
        )
        
        assert config.qk_mqa_dim == 128
        assert config.kv_lora_rank == 256
        assert config.cal_dataset == "custom_dataset"

    def test_to_dict(self):
        """Test configuration to dictionary conversion"""
        config = TransMLAConfig(qk_mqa_dim=128)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['qk_mqa_dim'] == 128
        assert 'freqfold' in config_dict
        assert 'collapse' in config_dict


class TestIntegration:
    """Integration tests for the complete workflow"""

    def test_end_to_end_workflow(self):
        """Test complete conversion workflow"""
        model = SimpleModel()
        tokenizer = Mock()
        
        # Direct conversion workflow
        with patch.object(TransMLA, '_convert_trans_mla') as mock_convert:
            mock_convert.return_value = model
            
            result = TransMLA.convert(
                model, 
                tokenizer=tokenizer
            )
            
            assert result == model
            mock_convert.assert_called_once()

    def test_factory_workflow(self):
        """Test using convert_model function for TransMLA conversion"""
        model = SimpleModel()
        tokenizer = Mock()
        
        # Use convert_model function for TransMLA conversion
        with patch.object(TransMLA, 'convert') as mock_convert:
            mock_convert.return_value = model
            
            result = convert_model(
                model,
                'trans-mla',
                tokenizer=tokenizer
            )
            
            assert result == model
            mock_convert.assert_called_once_with(model, tokenizer=tokenizer)


if __name__ == '__main__':
    pytest.main([__file__])
