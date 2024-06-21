from train.train import trainer
if __name__=='__main__':
    
    #config
    CONFIGURATION= {'LEARNING_RATE': 0.001,
                    'EPOCHS': 50,
                    'ADAM_BETAS': (0.9, 0.999),
                    'ADAM_EPS': 1e-07,
                    'SEED': 42,
                    'BATCH_SIZE': 16,
                    'LINEAR_LR': False,
                    'DROPOUTS': 0.3,
                    'L2_REGULARISATION': 0.001
                    }

    trainer(CONFIGURATION)
