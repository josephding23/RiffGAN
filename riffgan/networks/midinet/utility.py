import torch


def conv_cond_concat(x, y):
    x_shapes = x.shape
    y_shapes = y.shape
    y2 = y.expand(x_shapes[0],y_shapes[1],x_shapes[2],x_shapes[3])

    return torch.cat((x, y2),1)


def conv_prev_concat(x, y):
    x_shapes = x.shape
    y_shapes = y.shape
    if x_shapes[2:] == y_shapes[2:]:
        y2 = y.expand(x_shapes[0],y_shapes[1],x_shapes[2],x_shapes[3])

        return torch.cat((x, y2),1)

    else:
        print(x_shapes[2:])
        print(y_shapes[2:])