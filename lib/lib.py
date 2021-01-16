
from pathlib import Path
from fastai.vision import models, Learner, get_transforms, F, NormType, SplitFuncOrIdxList
from fastai.vision.learner import cnn_config, create_body, to_device
from fastai.vision.data import ImageImageList, ImageDataBunch, imagenet_stats, DataBunch, Callable, Optional, Tuple, Any
from fastai.layers import SequentialEx, nn
from fastai.callbacks.hooks import model_sizes, hook_outputs, dummy_eval, Hook

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from fastai.vision import *

from abc import ABC, abstractmethod
from PIL import Image
import PIL
import cv2

class IFilter(ABC):
    @abstractmethod
    def filter(
        self, orig_image: Image, filtered_image: Image, render_factor: int
    ) -> Image:
        pass

class ModelImageVisualizer:
    def __init__(self, filter: IFilter):
        self.filter = filter

    def _clean_mem(self):
        torch.cuda.empty_cache()
        # gc.collect()

    def _open_pil_image(self, path: Path) -> Image:
        return PIL.Image.open(path).convert('RGB')

    def _get_image_from_url(self, url: str) -> Image:
        response = requests.get(url, timeout=30, headers={'Accept': '*/*;q=0.8'})
        img = PIL.Image.open(BytesIO(response.content)).convert('RGB')
        return img

    def plot_transformed_image_from_url(
        self,
        url: str,
        path: str = 'test_images/image.png',
        figsize: (int, int) = (20, 20),
        render_factor: int = None,
        
        display_render_factor: bool = False,
        compare: bool = False,
        post_process: bool = True,
        watermarked: bool = True,
    ) -> Path:
        img = self._get_image_from_url(url)
        img.save(path)
        return self.plot_transformed_image(
            path=path,
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
            compare=compare,
            post_process = post_process,
            watermarked=watermarked,
        )

    def plot_transformed_image(
        self,
        path: str,
        out_path: str,
        figsize: (int, int) = (20, 20),
        render_factor: int = None,
        display_render_factor: bool = False,
        compare: bool = False,
        post_process: bool = True,
        watermarked: bool = True,
    ) -> Path:
        path = Path(path)
        result = self.get_transformed_image(
            path, render_factor, post_process=post_process,watermarked=watermarked
        )
        orig = self._open_pil_image(path)
        if compare:
            self._plot_comparison(
                figsize, render_factor, display_render_factor, orig, result
            )
        else:
            self._plot_solo(figsize, render_factor, display_render_factor, result)

        orig.close()
        result_path = self._save_result_image(out_path, result)
        result.close()
        return result_path

    def _plot_comparison(
        self,
        figsize: (int, int),
        render_factor: int,
        display_render_factor: bool,
        orig: Image,
        result: Image,
    ):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_image(
            orig,
            axes=axes[0],
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=False,
        )
        self._plot_image(
            result,
            axes=axes[1],
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
        )

    def _plot_solo(
        self,
        figsize: (int, int),
        render_factor: int,
        display_render_factor: bool,
        result: Image,
    ):
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        self._plot_image(
            result,
            axes=axes,
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
        )

    def _save_result_image(self, out_path: Path, image: Image) -> Path:
        image.save(out_path)
        return out_path

    def get_transformed_image(
        self, path: Path, render_factor: int = None, post_process: bool = True,
        watermarked: bool = True,
    ) -> Image:
        self._clean_mem()
        orig_image = self._open_pil_image(path)
        filtered_image = self.filter.filter(
            orig_image, orig_image, render_factor=render_factor,post_process=post_process
        )

        if watermarked:
            return get_watermarked(filtered_image)

        return filtered_image

    def _plot_image(
        self,
        image: Image,
        render_factor: int,
        axes: Axes = None,
        figsize=(20, 20),
        display_render_factor = False,
    ):
        if axes is None:
            _, axes = plt.subplots(figsize=figsize)
        axes.imshow(np.asarray(image) / 255)
        axes.axis('off')
        if render_factor is not None and display_render_factor:
            plt.text(
                10,
                10,
                'render_factor: ' + str(render_factor),
                color='white',
                backgroundcolor='black',
            )

    def _get_num_rows_columns(self, num_images: int, max_columns: int) -> (int, int):
        columns = min(num_images, max_columns)
        rows = num_images // columns
        rows = rows if rows * columns == num_images else rows + 1
        return rows, columns

class DynamicUnetWide(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        blur: bool = False,
        blur_final=True,
        self_attention: bool = False,
        y_range: Optional[Tuple[float, float]] = None,
        last_cross: bool = True,
        bottle: bool = False,
        norm_type: Optional[NormType] = NormType.Batch,
        nf_factor: int = 1,
        **kwargs
    ):

        nf = 512 * nf_factor
        extra_bn = norm_type == NormType.Spectral
        imsize = (256, 256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(
            custom_conv_layer(
                ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs
            ),
            custom_conv_layer(
                ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs
            ),
        ).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sfs_idxs) - 3)

            n_out = nf if not_final else nf // 2

            unet_block = UnetBlockWide(
                up_in_c,
                x_in_c,
                n_out,
                self.sfs[i],
                final_div=not_final,
                blur=blur,
                self_attention=sa,
                norm_type=norm_type,
                extra_bn=extra_bn,
                **kwargs
            ).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, norm_type=norm_type, **kwargs))
        layers += [
            custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type)
        ]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()

class CustomPixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(
        self,
        ni: int,
        nf: int = None,
        scale: int = 2,
        blur: bool = False,
        leaky: float = None,
        **kwargs
    ):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = custom_conv_layer(
            ni, nf * (scale ** 2), ks=1, use_activ=False, **kwargs
        )
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x

class UnetBlockWide(nn.Module):
    "A quasi-UNet blockSequentialEx, using `PixelShuffle_ICNR upsampling`."

    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        n_out: int,
        hook: Hook,
        final_div: bool = True,
        blur: bool = False,
        leaky: float = None,
        self_attention: bool = False,
        **kwargs
    ):
        super().__init__()
        self.hook = hook
        up_out = x_out = n_out // 2
        self.shuf = CustomPixelShuffle_ICNR(
            up_in_c, up_out, blur=blur, leaky=leaky, **kwargs
        )
        self.bn = batchnorm_2d(x_in_c)
        ni = up_out + x_in_c
        self.conv = custom_conv_layer(
            ni, x_out, leaky=leaky, self_attention=self_attention, **kwargs
        )
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv(cat_x)

class BaseFilter(IFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__()
        self.learn = learn
        self.device = next(self.learn.model.parameters()).device
        self.norm, self.denorm = normalize_funcs(*stats)

    def _transform(self, image: Image) -> Image:
        return image

    def _scale_to_square(self, orig: Image, targ: int) -> Image:
        # a simple stretch to fit a square really makes a big difference in rendering quality/consistency.
        # I've tried padding to the square as well (reflect, symetric, constant, etc).  Not as good!
        targ_sz = (targ, targ)
        return orig.resize(targ_sz, resample=PIL.Image.BILINEAR)

    def _get_model_ready_image(self, orig: Image, sz: int) -> Image:
        result = self._scale_to_square(orig, sz)
        result = self._transform(result)
        return result

    def _model_process(self, orig: Image, sz: int) -> Image:
        model_image = self._get_model_ready_image(orig, sz)
        x = pil2tensor(model_image, np.float32)
        x = x.to(self.device)
        x.div_(255)
        x, y = self.norm((x, x), do_x=True)
        
        try:
            result = self.learn.pred_batch(
                ds_type=DatasetType.Valid, batch=(x[None], y[None]), reconstruct=True
            )
        except RuntimeError as rerr:
            if 'memory' not in str(rerr):
                raise rerr
            print('Warning: render_factor was set too high, and out of memory error resulted. Returning original image.')
            return model_image
            
        out = result[0]
        out = self.denorm(out.px, do_x=False)
        out = image2np(out * 255).astype(np.uint8)
        return Image.fromarray(out)

    def _unsquare(self, image: Image, orig: Image) -> Image:
        targ_sz = orig.size
        image = image.resize(targ_sz, resample=PIL.Image.BILINEAR)
        return image

class MasterFilter(BaseFilter):
    def __init__(self, filters: [IFilter], render_factor: int):
        self.filters = filters
        self.render_factor = render_factor

    def filter(
        self, orig_image: Image, filtered_image: Image, render_factor: int = None, post_process: bool = True) -> Image:
        render_factor = self.render_factor if render_factor is None else render_factor
        for filter in self.filters:
            filtered_image = filter.filter(orig_image, filtered_image, render_factor, post_process)

        return filtered_image

class ColorizerFilter(BaseFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__(learn=learn, stats=stats)
        self.render_base = 16

    def filter(
        self, orig_image: Image, filtered_image: Image, render_factor: int, post_process: bool = True) -> Image:
        render_sz = render_factor * self.render_base
        model_image = self._model_process(orig=filtered_image, sz=render_sz)
        raw_color = self._unsquare(model_image, orig_image)

        if post_process:
            return self._post_process(raw_color, orig_image)
        else:
            return raw_color

    def _transform(self, image: Image) -> Image:
        return image.convert('LA').convert('RGB')

    # This takes advantage of the fact that human eyes are much less sensitive to
    # imperfections in chrominance compared to luminance.  This means we can
    # save a lot on memory and processing in the model, yet get a great high
    # resolution result at the end.  This is primarily intended just for
    # inference
    def _post_process(self, raw_color: Image, orig: Image) -> Image:
        color_np = np.asarray(raw_color)
        orig_np = np.asarray(orig)
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_BGR2YUV)
        # do a black and white transform first to get better luminance values
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_BGR2YUV)
        hires = np.copy(orig_yuv)
        hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        final = cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)
        final = Image.fromarray(final)
        return final

def get_watermarked(pil_image: Image) -> Image:
    try:
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        (h, w) = image.shape[:2]
        image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
        pct = 0.05
        full_watermark = cv2.imread(
            './resource_images/watermark.png', cv2.IMREAD_UNCHANGED
        )
        (fwH, fwW) = full_watermark.shape[:2]
        wH = int(pct * h)
        wW = int((pct * h / fwH) * fwW)
        watermark = cv2.resize(full_watermark, (wH, wW), interpolation=cv2.INTER_AREA)
        overlay = np.zeros((h, w, 4), dtype="uint8")
        (wH, wW) = watermark.shape[:2]
        overlay[h - wH - 10 : h - 10, 10 : 10 + wW] = watermark
        # blend the two images together using transparent overlays
        output = image.copy()
        cv2.addWeighted(overlay, 0.5, output, 1.0, 0, output)
        rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(rgb_image)
        return final_image
    except:
        # Don't want this to crash everything, so let's just not watermark the image for now.
        return pil_image

def custom_conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    padding: int = None,
    bias: bool = None,
    is_1d: bool = False,
    norm_type: Optional[NormType] = NormType.Batch,
    use_activ: bool = True,
    leaky: float = None,
    transpose: bool = False,
    init: Callable = nn.init.kaiming_normal_,
    self_attention: bool = False,
    extra_bn: bool = False,
):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn == True
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(
        conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
        init,
    )
    if norm_type == NormType.Weight:
        conv = weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)

def _get_sfs_idxs(sizes: Sizes) -> List[int]:
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(
        np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0]
    )
    if feature_szs[0] != feature_szs[1]:
        sfs_idxs = [0] + sfs_idxs
    return sfs_idxs

def get_image_colorizer(render_factor: int = 35) -> ModelImageVisualizer:
    return get_stable_image_colorizer(render_factor=render_factor)

def get_stable_image_colorizer(
    root_folder: Path = Path('data'),
    weights_name: str = 'ColorizeStable_gen',
    render_factor: int = 35
) -> ModelImageVisualizer:
    learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr)
    return vis

def unet_learner_wide(
    data: DataBunch,
    arch: Callable,
    pretrained: bool = True,
    blur_final: bool = True,
    norm_type: Optional[NormType] = NormType,
    split_on: Optional[SplitFuncOrIdxList] = None,
    blur: bool = False,
    self_attention: bool = False,
    y_range: Optional[Tuple[float, float]] = None,
    last_cross: bool = True,
    bottle: bool = False,
    nf_factor: int = 1,
    **kwargs: Any
) -> Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)
    model = to_device(
        DynamicUnetWide(
            body,
            n_classes=data.c,
            blur=blur,
            blur_final=blur_final,
            self_attention=self_attention,
            y_range=y_range,
            norm_type=norm_type,
            last_cross=last_cross,
            bottle=bottle,
            nf_factor=nf_factor,
        ),
        data.device,
    )
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained:
        learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn


def gen_learner_wide(
    data: ImageDataBunch, gen_loss, arch=models.resnet101, nf_factor: int = 2
) -> Learner:
    return unet_learner_wide(
        data,
        arch=arch,
        wd=1e-3,
        blur=True,
        norm_type=NormType.Spectral,
        self_attention=True,
        y_range=(-3.0, 3.0),
        loss_func=gen_loss,
        nf_factor=nf_factor,
    )

def gen_inference_wide(
    root_folder: Path, weights_name: str, nf_factor: int = 2, arch=models.resnet101
    ) -> Learner:
    data = get_dummy_databunch()
    learn = gen_learner_wide(
        data=data, gen_loss=F.l1_loss, nf_factor=nf_factor, arch=arch
    )
    learn.path = root_folder
    learn.load(weights_name)
    learn.model.eval()
    return learn

def get_colorize_data(
    sz: int,
    bs: int,
    crappy_path: Path,
    good_path: Path,
    random_seed: int = None,
    keep_pct: float = 1.0,
    num_workers: int = 8,
    stats: tuple = imagenet_stats,
    xtra_tfms=[],
) -> ImageDataBunch:
    
    src = (
        ImageImageList.from_folder(crappy_path, convert_mode='RGB')
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .split_by_rand_pct(0.1, seed=random_seed)
    )

    data = (
        src.label_from_func(lambda x: good_path / x.relative_to(crappy_path))
        .transform(
            get_transforms(
                max_zoom=1.2, max_lighting=0.5, max_warp=0.25, xtra_tfms=xtra_tfms
            ),
            size=sz,
            tfm_y=True,
        )
        .databunch(bs=bs, num_workers=num_workers, no_check=True)
        .normalize(stats, do_y=True)
    )

    data.c = 3
    return data


def get_dummy_databunch() -> ImageDataBunch:
    path = Path('./dummy/')
    return get_colorize_data(
        sz=1, bs=1, crappy_path=path, good_path=path, keep_pct=0.001
    )

