import DeepFried2 as df


class Flatten(df.Module):
    def symb_forward(self, symb_in):
        return symb_in.flatten(2)


def mknet_gpu(*outlayers):
    return df.Sequential(                          #     3@46
        df.SpatialConvolutionCUDNN( 3, 24, 3, 3),  # -> 24@44
        df.BatchNormalization(24),
        df.ReLU(),
        df.SpatialConvolutionCUDNN(24, 24, 3, 3),  # -> 24@42
        df.BatchNormalization(24),
        df.SpatialMaxPoolingCUDNN(2, 2),           # -> 24@21
        df.ReLU(),
        df.SpatialConvolutionCUDNN(24, 48, 3, 3),  # -> 48@19
        df.BatchNormalization(48),
        df.ReLU(),
        df.SpatialConvolutionCUDNN(48, 48, 3, 3),  # -> 48@17
        df.BatchNormalization(48),
        df.SpatialMaxPooling(2, 2),                # -> 48@9
        df.ReLU(),
        df.SpatialConvolutionCUDNN(48, 64, 3, 3),  # -> 48@7
        df.BatchNormalization(64),
        df.ReLU(),
        df.SpatialConvolutionCUDNN(64, 64, 3, 3),  # -> 48@5
        df.BatchNormalization(64),
        df.ReLU(),
        df.Dropout(0.2),
        Flatten(),
        df.Linear(64*5*5, 512),
        df.ReLU(),
        df.Dropout(0.5),
        *outlayers
    )


def mknet_cpu(*outlayers):
    return df.Sequential(                          #     3@46
        df.SpatialConvolution( 3, 24, 3, 3),  # -> 24@44
        df.BatchNormalization(24),
        df.ReLU(),
        df.SpatialConvolution(24, 24, 3, 3),  # -> 24@42
        df.BatchNormalization(24),
        df.SpatialMaxPooling(2, 2),           # -> 24@21
        df.ReLU(),
        df.SpatialConvolution(24, 48, 3, 3),  # -> 48@19
        df.BatchNormalization(48),
        df.ReLU(),
        df.SpatialConvolution(48, 48, 3, 3),  # -> 48@17
        df.BatchNormalization(48),
        df.SpatialMaxPooling(2, 2),                # -> 48@9
        df.ReLU(),
        df.SpatialConvolution(48, 64, 3, 3),  # -> 48@7
        df.BatchNormalization(64),
        df.ReLU(),
        df.SpatialConvolution(64, 64, 3, 3),  # -> 48@5
        df.BatchNormalization(64),
        df.ReLU(),
        df.Dropout(0.2),
        Flatten(),
        df.Linear(64*5*5, 512),
        df.ReLU(),
        df.Dropout(0.5),
        *outlayers
    )