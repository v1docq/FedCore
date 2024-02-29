from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedcore.tools.benchmark import CompressionBenchmark

benchmark_setup = {
    'fedcore_setup':
        {
            'initial_assumption': PipelineBuilder().add_node('post_training_quant'),
            'framework_config':
            {
                'dtype': "int8",
                'opset_version': 16,
                'quant_format': "QDQ",  # or "QLinear"
                'input_names': ["input"],
                'output_names': ["output"],
                'dynamic_axes': {'input': [0], 'output': [0]}
            }
        }
}
if __name__ == '__main__':
    bench = CompressionBenchmark(benchmark_setup)
    compressed_model, results = bench.run('quantisation', 'ONNX')
