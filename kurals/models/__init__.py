from .kuralsnet import KuRALSNet
from .kuralsnet_npu_seg import KuRALSNetNPUSeg
from .kuralsnet_woaspp import KuRALSNet_WoASPP
from .kuralsnet_ada import KuRALSNet_ADA
from .kuralsnet_pkc import KuRALSNet_PKC
from .kuralsnet_adapkctheta import KuRALSNet_AdaPKCTheta
try:
    # Requires the optional compiled correlation_cuda extension (README install
    # step 4). Guarded so the rest of kurals.models stays importable without it.
    from .kuralsnet_adapkcxi import KuRALSNet_AdaPKCXi
except ImportError:
    KuRALSNet_AdaPKCXi = None

from .fcn8s import FCN8s
from .unet import UNet
from .deeplabv3plus import deeplabv3plus_resnet101
from .hrnet import HRNet
from .rssnet import RSSNet

from .segformer import SegFormer
from .swin_transformer.encoder_decoder import EncoderDecoder as Swin