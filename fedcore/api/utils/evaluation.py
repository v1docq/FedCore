def evaluate_original_model(fedcore_compressor, input_data):
    original_prediction = fedcore_compressor.predict(input_data, output_mode="default")
    original_output = original_prediction.predict
    original_model = fedcore_compressor.original_model
    original_quality_metrics = fedcore_compressor.evaluate_metric(
        predicton=original_output, target=fedcore_compressor.target
    )
    original_inference_metrics = fedcore_compressor.evaluate_metric(
        predicton=original_output,
        target=fedcore_compressor.target,
        metric_type="original_computational",
    )
    return dict(
        original_model=original_model,
        quality_metrics=original_quality_metrics,
        inference_metrics=original_inference_metrics,
    )


def evaluate_optimised_model(fedcore_compressor, input_data):
    low_rank_prediction = fedcore_compressor.predict(input_data, output_mode="compress")
    low_rank_output = low_rank_prediction.predict
    low_rank_model = fedcore_compressor.optimised_model
    low_rank_quality_metrics = fedcore_compressor.evaluate_metric(
        predicton=low_rank_output, target=fedcore_compressor.target
    )
    low_rank_inference_metrics = fedcore_compressor.evaluate_metric(
        predicton=low_rank_output,
        target=fedcore_compressor.target,
        metric_type="optimised_computational",
    )
    return dict(
        optimised_model=low_rank_model,
        quality_metrics=low_rank_quality_metrics,
        inference_metrics=low_rank_inference_metrics,
    )
