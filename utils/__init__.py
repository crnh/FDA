import torch
import numpy as np
from functools import lru_cache

####################################################################################################################
### Old code ###


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2

    fft_amp = fft_im[:, :, :, :, 0] ** 2 + fft_im[:, :, :, :, 1] ** 2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)  # get b
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]  # top left
    amp_src[:, :, 0:b, w - b : w] = amp_trg[:, :, 0:b, w - b : w]  # top right
    amp_src[:, :, h - b : h, 0:b] = amp_trg[:, :, h - b : h, 0:b]  # bottom left
    amp_src[:, :, h - b : h, w - b : w] = amp_trg[
        :, :, h - b : h, w - b : w
    ]  # bottom right
    return amp_src


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img
    # print(src_img.shape)

    img_ndim = src_img.dim()

    # get fft of both source and target
    fft_src = torch.fft.fftn(src_img.clone(), dim=(img_ndim - 2, img_ndim - 1))
    fft_trg = torch.fft.fftn(trg_img.clone(), dim=(img_ndim - 2, img_ndim - 1))

    # Change from complex numbers to additional dimension for real and imaginary part
    fft_src = torch.view_as_real(fft_src)
    fft_trg = torch.view_as_real(fft_trg)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float)
    fft_src_[:, :, :, :, 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:, :, :, :, 1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # Change back from additional dimension for real and imaginary part to complex numbers
    fft_src_ = torch.view_as_complex(fft_src_)

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.ifftn(
        fft_src_, dim=(img_ndim - 2, img_ndim - 1), s=(imgH, imgW)
    )

    # print(f"Source in target image data type: {src_in_trg.dtype}")
    # print(f"Single value from new image: {src_in_trg[0, 0, 100, 100]}")

    # Convert to real-valued tensor
    return torch.real(src_in_trg)


def FDA_source_to_target_np(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img  # .cpu().numpy()
    trg_img_np = trg_img  # .cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(trg_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


######################################################################################################
#######################################Homemade functions#############################################
######################################################################################################


def centered_fft(image):
    fft = torch.fft.fftshift(torch.fft.fftn(image, dim=(-2, -1)), dim=(-2, -1))

    return fft


def centered_ifft(fft, s=None):
    image = torch.fft.ifftn(torch.fft.ifftshift(fft, dim=(-2, -1)), dim=(-2, -1), s=s)

    return image


@lru_cache(maxsize=None)
def circular_mask(h, w, beta_pixel):
    """
    Calculates a circular mask with height `h` and width `w`
    """
    r_square = beta_pixel ** 2 / np.pi

    center_height = h // 2
    center_width = w // 2

    # make circle mask
    H, W = torch.meshgrid(torch.arange(h), torch.arange(w))
    mask = ((H - center_height) ** 2 + (W - center_width) ** 2) <= r_square

    return mask


@lru_cache(maxsize=None)
def gaussian_mask(h, w, beta_pixel):
    """
    Calculates a gaussian mask with height `h` and width `w`
    """
    x, y = torch.meshgrid(torch.arange(h), torch.arange(w))
    mu_x = h // 2
    mu_y = w // 2

    sigma_square = beta_pixel ** 2 / (2 * torch.pi)

    # Generate mask with maximum value of 1 at mu_x, mu_y
    mask = torch.exp(-((x - mu_x) ** 2) / sigma_square / 2) * torch.exp(
        -((y - mu_y) ** 2) / sigma_square / 2
    )

    return mask


def shape_low_freq_mutate(amp_src, amp_trg, beta, mask_type="circle"):
    """
    Exchange amplitude of the center of the target image (amp_trg) with that for the
    source image (amp_src). Possible shapes 'rectangle', 'circle' or 'gaussian'

    """
    _, _, h, w = amp_src.size()
    # beta_pixel = int(
    #     (min(h, w) * beta) ** 2
    # )  # specificity the number of pixels in a square as depicted in the paper

    beta_pixel = int((min(h, w) * beta))

    if mask_type == "rectangle":
        b = (np.floor(np.amin((h, w)) * beta)).astype(int)  # get b

        amp_src[
            :,
            :,
            int(h / 2) - int(b / 2) : int(h / 2) + int(b / 2),
            int(w / 2) - int(b / 2) : int(w / 2) + int(b / 2),
        ] = amp_trg[
            :,
            :,
            int(h / 2) - int(b / 2) : int(h / 2) + int(b / 2),
            int(w / 2) - int(b / 2) : int(w / 2) + int(b / 2),
        ]

        return amp_src

    if mask_type == "circle":
        # R = np.sqrt(beta_pixel / np.pi)
        # center_height = h // 2
        # center_width = w // 2

        # # make circle mask
        # H, W = torch.meshgrid(torch.arange(h), torch.arange(w))
        # mask = ((H - center_height) ** 2 + (W - center_width) ** 2) <= R ** 2

        mask = circular_mask(h, w, beta_pixel)

        # apply mask
        return amp_trg * mask + amp_src * ~mask

    if mask_type == "gaussian":
        # set constants for gauss mask
        # x, y = torch.meshgrid(torch.arange(h), torch.arange(w))
        # mu_x = h // 2
        # mu_y = w // 2
        # sigma = np.sqrt(beta_pixel / np.pi) * 1.5
        # norm_factor = 1 / ((sigma ** 2) * 2 * np.pi)

        # # fill gauss mask
        # mask = (
        #     beta_pixel
        #     * norm_factor
        #     * (
        #         torch.exp(
        #             -(
        #                 ((abs(x) - mu_x) ** 2 + (abs(y) - mu_y) ** 2)
        #                 / (2.0 * sigma ** 2)
        #             )
        #         )
        #     )
        # )
        # # make the inverse of the gauss mask
        # invers_mask = torch.full((h, w), torch.max(torch.max(mask))) - mask
        # # aplly  masks and mistery factor of 4.5 to scale to same intesnety range as other FDA shapes

        mask = gaussian_mask(h, w, beta_pixel)
        return amp_trg * mask + amp_src * (1 - mask)

    else:
        raise ValueError(
            "mask_type not recognised, please use 'rectangle', 'circle' or 'gaussian'"
        )

def shape_FDA_source_to_target(src_img, trg_img, beta=0.01, mask_type="circle"):

    img_ndim = src_img.dim()

    # get fft of both source and target
    # get fft of both source and target
    fft_src = centered_fft(src_img.clone())
    fft_trg = centered_fft(trg_img.clone())

    # Change from complex numbers to additional dimension for real and imaginary part
    fft_src = torch.view_as_real(fft_src)
    fft_trg = torch.view_as_real(fft_trg)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = shape_low_freq_mutate(amp_src.clone(), amp_trg.clone(), beta, mask_type)

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float)
    fft_src_[:, :, :, :, 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:, :, :, :, 1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # Change back from additional dimension for real and imaginary part to complex numbers
    fft_src_ = torch.view_as_complex(fft_src_)

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = centered_ifft(fft_src_, s=(imgH, imgW))

    # Convert to real-valued tensor
    return torch.real(src_in_trg)
