import importlib
from enum import Enum
from typing import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union
)

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes, Clip, Coordinates, Number
from starfish.core.util import click
from starfish.core.util.dtype import preserve_float_range
from ._base import FilterAlgorithmBase


class FunctionSource(Enum):
    def __init__(self, source: str, aliases: Optional[Mapping[str, str]] = None):
        self.source = source
        self.aliases = aliases or {}

    @property
    def module(self):
        return importlib.import_module(self.source)

    np = ("numpy", {'max': 'amax'})


class Reduce(FilterAlgorithmBase):
    """
    Reduces the dimensions of the ImageStack by applying a function along one or more axes.

    Parameters
    ----------
    dims : Axes
        one or more Axes to project over
    func : str
        Name of a function in the module specified by the `module` parameter to apply across the
        dimension(s) specified by dims.  The function is resolved by getattr(<module>, func), except
        in the cases of predefined aliases.  See :py:class:FunctionSource for more information about
        aliases.

        Some common examples for the np FunctionSource:
        amax: maximum intensity projection (applies np.amax)
        max: maximum intensity projection (this is an alias for amax and applies np.amax)
        mean: take the mean across the dim(s) (applies np.mean)
        sum: sum across the dim(s) (applies np.sum)
    module : FunctionSource
        Python module that serves as the source of the function.  It must be listed as one of the
        members of :py:class:FunctionSource.
    clip_method : Clip
        (Default Clip.CLIP) Controls the way that data are scaled to retain skimage dtype
        requirements that float data fall in [0, 1].
        Clip.CLIP: data above 1 are set to 1, and below 0 are set to 0
        Clip.SCALE_BY_IMAGE: data above 1 are scaled by the maximum value, with the maximum
        value calculated over the entire ImageStack

    See Also
    --------
    starfish.types.Axes

    """

    def __init__(
        self,
            dims: Iterable[Union[Axes, str]],
            func: str = "max",
            module: FunctionSource = FunctionSource.np,
            clip_method: Clip = Clip.CLIP
    ) -> None:

        self.dims = dims
        self.clip_method = clip_method

        function_source = module.module
        method_name = module.aliases.get(func, func)
        self.func = getattr(function_source, method_name)

    _DEFAULT_TESTING_PARAMETERS = {"dims": ['r'], "func": 'max'}

    def run(
            self,
            stack: ImageStack,
            *args,
    ) -> ImageStack:
        """Performs the dimension reduction with the specifed function

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack. Otherwise return the
            original stack.

        """

        # Apply the reducing function
        reduced = stack._data.reduce(self.func, dim=[Axes(dim).value for dim in self.dims])

        # Add the reduced dims back and align with the original stack
        reduced = reduced.expand_dims(tuple(Axes(dim).value for dim in self.dims))
        reduced = reduced.transpose(*stack.xarray.dims)

        if self.clip_method == Clip.CLIP:
            reduced = preserve_float_range(reduced, rescale=False)
        else:
            reduced = preserve_float_range(reduced, rescale=True)

        # Update the physical coordinates
        physical_coords: MutableMapping[Coordinates, Sequence[Number]] = {}
        for axis, coord in (
                (Axes.X, Coordinates.X),
                (Axes.Y, Coordinates.Y),
                (Axes.ZPLANE, Coordinates.Z)):
            if axis in self.dims:
                # this axis was projected out of existence.
                assert coord.value not in reduced.coords
                physical_coords[coord] = [np.average(stack._data.coords[coord.value])]
            else:
                physical_coords[coord] = reduced.coords[coord.value]
        reduced_stack = ImageStack.from_numpy(reduced.values, coordinates=physical_coords)

        return reduced_stack

    @staticmethod
    @click.command("Reduce")
    @click.option(
        "--dims",
        type=click.Choice(
            [Axes.ROUND.value, Axes.CH.value, Axes.ZPLANE.value, Axes.X.value, Axes.Y.value]
        ),
        multiple=True,
        help="The dimensions the Imagestack should max project over."
             "For multiple dimensions add multiple --dims. Ex."
             "--dims r --dims c")
    @click.option(
        "--func",
        type=str,
        help="The function to apply across dims."
    )
    @click.option(
        "--module",
        type=click.Choice([member.name for member in list(FunctionSource)]),
        multiple=False,
        help="Module to source the function from.",
        default=FunctionSource.np.name,
    )
    @click.option(
        "--clip-method", default=Clip.CLIP, type=Clip,
        help="method to constrain data to [0,1]. options: 'clip', 'scale_by_image'")
    @click.pass_context
    def _cli(ctx, dims, func, module, clip_method):
        ctx.obj["component"]._cli_run(ctx, Reduce(dims, func, FunctionSource[module], clip_method))
