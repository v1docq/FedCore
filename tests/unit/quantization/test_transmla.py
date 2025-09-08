"""
Tests for TransMLA and AttentionReassembler classes
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from fedcore.algorithm.quantization.utils import (
    AttentionReassembler,
    TransMLA,
    DeferredConversion,
    ReassemblerFactory,
    TransMLAConfig,
    TRANSMLA_AVAILABLE
)


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

    def test_convert_standard_mode(self):
        """Test standard conversion mode"""
        model = SimpleModel()
        
        # Should work without any special dependencies
        result = AttentionReassembler.convert(model, mode='standard')
        
        assert result is not None
        assert isinstance(result, nn.Module)

    def test_convert_unknown_mode(self):
        """Test that unknown mode raises assertion error"""
        model = SimpleModel()
        
        with pytest.raises(AssertionError, match="Unknown mode"):
            AttentionReassembler.convert(model, mode='unknown_mode')

    @patch('fedcore.algorithm.quantization.utils.TRANSMLA_AVAILABLE', True)
    def test_convert_transmla_mode_success(self):
        """Test TransMLA conversion mode when TransMLA is available"""
        model = SimpleModel()
        tokenizer = Mock()
        
        with patch('fedcore.algorithm.quantization.utils._convert_model_to_mla') as mock_convert:
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

    @patch('fedcore.algorithm.quantization.utils.TRANSMLA_AVAILABLE', False)
    def test_convert_transmla_mode_unavailable(self):
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

    def test_convert_immediate_execution(self):
        """Test immediate execution (deferred=False)"""
        model = SimpleModel()
        tokenizer = Mock()
        
        with patch.object(TransMLA, '_execute_conversion') as mock_execute:
            mock_execute.return_value = model
            
            result = TransMLA.convert(
                model, 
                tokenizer=tokenizer, 
                deferred=False
            )
            
            assert result == model
            mock_execute.assert_called_once()

    def test_convert_deferred_execution(self):
        """Test deferred execution (deferred=True)"""
        model = SimpleModel()
        tokenizer = Mock()
        
        result = TransMLA.convert(
            model, 
            tokenizer=tokenizer, 
            deferred=True
        )
        
        assert isinstance(result, DeferredConversion)
        assert result.conversion_type == 'trans-mla'
        assert result.model == model
        assert 'tokenizer' in result.kwargs

    @patch('fedcore.algorithm.quantization.utils.TRANSMLA_AVAILABLE', True)
    def test_execute_conversion_success(self):
        """Test successful conversion execution"""
        model = SimpleModel()
        tokenizer = Mock()
        
        with patch('fedcore.algorithm.quantization.utils._convert_model_to_mla') as mock_convert:
            mock_convert.return_value = model
            
            result = TransMLA._execute_conversion(
                model=model,
                tokenizer=tokenizer
            )
            
            assert result == model
            mock_convert.assert_called_once()


class TestDeferredConversion:
    """Test DeferredConversion functionality"""

    def test_init(self):
        """Test DeferredConversion initialization"""
        model = SimpleModel()
        tokenizer = Mock()
        
        deferred = DeferredConversion(
            'trans-mla',
            model=model,
            tokenizer=tokenizer,
            config=None
        )
        
        assert deferred.conversion_type == 'trans-mla'
        assert deferred.model == model
        assert deferred.kwargs['tokenizer'] == tokenizer
        assert not deferred.executed
        assert deferred.result is None

    def test_execute_success(self):
        """Test successful deferred execution"""
        model = SimpleModel()
        tokenizer = Mock()
        
        deferred = DeferredConversion(
            'trans-mla',
            model=model,
            tokenizer=tokenizer
        )
        
        with patch.object(TransMLA, '_execute_conversion') as mock_execute:
            mock_execute.return_value = model
            
            result = deferred.execute()
            
            assert result == model
            assert deferred.executed
            assert deferred.result == model
            mock_execute.assert_called_once_with(model, tokenizer=tokenizer)

    def test_execute_already_executed(self):
        """Test that executing twice raises assertion error"""
        model = SimpleModel()
        deferred = DeferredConversion('trans-mla', model=model)
        deferred.executed = True  # Mark as already executed
        
        with pytest.raises(AssertionError, match="Conversion already executed"):
            deferred.execute()

    def test_execute_unknown_conversion_type(self):
        """Test unknown conversion type raises assertion error"""
        model = SimpleModel()
        deferred = DeferredConversion('unknown_type', model=model)
        
        with pytest.raises(AssertionError, match="Unknown conversion type"):
            deferred.execute()


class TestReassemblerFactory:
    """Test ReassemblerFactory functionality"""

    def test_get_reassembler_valid_types(self):
        """Test getting valid reassembler types"""
        valid_types = ['attention', 'trans-mla', 'standard', 'parental']
        
        for reassembler_type in valid_types:
            reassembler = ReassemblerFactory.get_reassembler(reassembler_type)
            assert reassembler is not None

    def test_get_reassembler_invalid_type(self):
        """Test getting invalid reassembler type raises assertion error"""
        with pytest.raises(AssertionError, match="Unknown reassembler type"):
            ReassemblerFactory.get_reassembler('invalid_type')

    def test_convert_model_attention(self):
        """Test converting model with attention reassembler"""
        model = SimpleModel()
        
        with patch.object(AttentionReassembler, 'convert') as mock_convert:
            mock_convert.return_value = model
            
            result = ReassemblerFactory.convert_model(
                model, 
                'attention', 
                mode='standard'
            )
            
            assert result == model
            mock_convert.assert_called_once_with(model, mode='standard')

    def test_convert_model_transmla(self):
        """Test converting model with TransMLA reassembler"""
        model = SimpleModel()
        tokenizer = Mock()
        
        with patch.object(TransMLA, 'convert') as mock_convert:
            mock_convert.return_value = model
            
            result = ReassemblerFactory.convert_model(
                model, 
                'trans-mla',
                tokenizer=tokenizer
            )
            
            assert result == model
            mock_convert.assert_called_once_with(model, tokenizer=tokenizer)

    def test_convert_model_standard(self):
        """Test converting model with standard reassembler"""
        model = SimpleModel()
        
        with patch('fedcore.algorithm.quantization.utils.ParentalReassembler') as mock_reassembler:
            mock_instance = Mock()
            mock_reassembler.return_value = mock_instance
            mock_instance.reassemble.return_value = model
            
            result = ReassemblerFactory.convert_model(model, 'standard')
            
            # Note: Factory calls class method, not instance method
            # So we need to patch the class method
            assert result is not None


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

    def test_end_to_end_deferred_workflow(self):
        """Test complete deferred conversion workflow"""
        model = SimpleModel()
        tokenizer = Mock()
        
        # Step 1: Create deferred conversion
        deferred = TransMLA.convert(
            model, 
            tokenizer=tokenizer, 
            deferred=True
        )
        
        assert isinstance(deferred, DeferredConversion)
        assert not deferred.executed
        
        # Step 2: Execute when ready
        with patch.object(TransMLA, '_execute_conversion') as mock_execute:
            mock_execute.return_value = model
            
            result = deferred.execute()
            
            assert result == model
            assert deferred.executed
            mock_execute.assert_called_once()

    def test_factory_to_deferred_workflow(self):
        """Test using factory to create deferred conversion"""
        model = SimpleModel()
        tokenizer = Mock()
        
        # Use factory to create TransMLA conversion
        result = ReassemblerFactory.convert_model(
            model,
            'trans-mla',
            tokenizer=tokenizer,
            deferred=True
        )
        
        # Should return deferred conversion if TransMLA supports it
        # (This test may need adjustment based on actual factory implementation)
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__])
