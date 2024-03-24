from .modules import Generator, StyleGanGenerator, Discriminator
from .sinkhorn import SinkhornDistance
from .stylegan2 import StyledGenerator2, StyleDiscriminator

__all__ = [Generator, StyleGanGenerator, Discriminator, SinkhornDistance,StyledGenerator2,StyleDiscriminator]