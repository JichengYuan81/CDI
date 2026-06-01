def rotation2():
    def _transform(images):
        size = images.shape[1:]
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in [0, 2]], 1).view(-1, *size)

    return _transform, 2

def rot_color_perm12():
    def _transform(images):
        size = images.shape[1:]
        out = []
        for k in range(4):
            x = torch.rot90(images, k, (2, 3))
            out.append(x)
            out.append(torch.stack([x[:, 1, :, :], x[:, 2, :, :], x[:, 0, :, :]], 1))
            out.append(torch.stack([x[:, 2, :, :], x[:, 0, :, :], x[:, 1, :, :]], 1))
        return torch.stack(out, 1).view(-1, *size).contiguous()

    return _transform, 12
