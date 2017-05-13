import numpy as np
np.random.seed(3318)

def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)
    #return arr


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data


if __name__ == '__main__':
    # sample mean and var
    times = 1000
    N = 100
    meanAndVar  = []
    for dim in range(1, 61):
        sampleData = []
        print("dim =", dim)
        for t in range(times):
            print("\rt = " + str(t + 1) + "/" + str(times), end = "")
            layerDims = [np.random.randint(60, 80), 100]
            genData = gen_data(dim, layerDims, N).reshape(-1)
            sampleData.append(genData)
        sampleData = np.array(sampleData).reshape(-1, 100)
        print("\nshape, mean, var |", sampleData.shape, np.mean(sampleData), np.var(sampleData) )
        meanAndVar.append([np.mean(sampleData), np.var(sampleData)])
    np.save( "mean_and_var", np.array(meanAndVar) )

