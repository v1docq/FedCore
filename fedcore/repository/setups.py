PTQ_1 = {'compression_task': 'composite_compression',
       'need_pretrain': False,
                    'common': dict(save_each=10, passing=False),
                    'model_params': dict(
                                    training_model=dict(
                                             epochs=99,
                                         ),
                                    post_training_quant=dict(
                                        epochs=1
                                    )                    
                    ),  
                    'initial_assumption': [
                        'training_model',
                        'post_training_quant'
                    ]}
QAT_1 = {'compression_task': 'composite_compression',
       'need_pretrain': False,
                    'common': dict(save_each=10, passing=False),
                    'model_params': dict(
                                    training_model=dict(
                                             epochs=50,
                                         ),
                                    training_aware_quant=dict(
                                        epochs=50
                                    )                    
                                    ),  
                    'initial_assumption': [
                        'training_model',
                        'training_aware_quant'
                    ]}
