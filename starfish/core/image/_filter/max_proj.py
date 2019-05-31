from typing import Iterable

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes
from starfish.core.util import click
from ._base import FilterAlgorithmBase


class MaxProject(FilterAlgorithmBase):
    """
    Creates a maximum projection over one or more axis of the image tensor

    Parameters
    ----------
    dims : Iterable[Axes]
        one or more Axes to project over

    See Also
    --------
    starfish.types.Axes

    """

    def __init__(self, dims: Iterable[Axes]) -> None:
        self.dims = dims

    _DEFAULT_TESTING_PARAMETERS = {"dims": 'r'}

    def run(
            self,
            stack: ImageStack,
            verbose: bool = False,
            *args,
    ) -> ImageStack:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        verbose : bool
            if True, report on filtering progress (default = False)

        Returns
        -------
        ImageStack :
            The max projection of an image across one or more axis.

        """
        return stack._max_proj(*self.dims)

    @staticmethod
    @click.command("MaxProject")
    @click.option(
        "--dims",
        type=click.Choice(
            [Axes.ROUND.value, Axes.CH.value, Axes.ZPLANE.value, Axes.X.value, Axes.Y.value]
        ),
        multiple=True,
        help="The dimensions the Imagestack should max project over."
             "For multiple dimensions add multiple --dims. Ex."
             "--dims r --dims c")
    @click.pass_context
    def _cli(ctx, dims):
        formatted_dims = [Axes(dim) for dim in dims]
        ctx.obj["component"]._cli_run(ctx, MaxProject(formatted_dims))
