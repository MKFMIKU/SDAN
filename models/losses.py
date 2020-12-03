import torch


def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


def smoothness_loss(flow):
    def charbonnier(x, alpha=0.25, epsilon=1.e-9):
        return torch.pow(torch.pow(x, 2) + epsilon ** 2, alpha)

    b, c, h, w = flow.size()
    v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
    h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
    s_loss = charbonnier(flow - v_translated) + charbonnier(flow - h_translated)
    s_loss = torch.sum(s_loss, dim=1) / 2

    return torch.sum(s_loss) / b


def gradient_x(img):
    return img[:, :, :, :-1]-img[:, :, :, 1:]


def gradient_y(img):
    return img[:, :, :-1, :]-img[:, :, 1:, :]


def smooth_loss(depth, image):
    gradient_depth_x = gradient_x(depth)  # (TODO)shape: bs,1,h,w
    gradient_depth_y = gradient_y(depth)

    gradient_img_x = gradient_x(image)  # (TODO)shape: bs,3,h,w
    gradient_img_y = gradient_y(image)

    exp_gradient_img_x = torch.exp(-torch.mean(torch.abs(gradient_img_x), 1, True)) # (TODO)shape: bs,1,h,w
    exp_gradient_img_y = torch.exp(-torch.mean(torch.abs(gradient_img_y), 1, True))

    smooth_x = gradient_depth_x*exp_gradient_img_x
    smooth_y = gradient_depth_y*exp_gradient_img_y

    return torch.mean(torch.abs(smooth_x))+torch.mean(torch.abs(smooth_y))


def flow_smooth_loss(flow, img):
    smoothness = 0
    for i in range(2):
        smoothness += smooth_loss(flow[:, i, :, :].unsqueeze(1), img)
    return smoothness/2
