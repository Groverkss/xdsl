from __future__ import annotations

from xdsl.dialects.builtin import (
    ArrayAttr,
    AffineMapAttr,
    StringAttr,
    ShapedType,
    AnyShapedType,
    AnyTensorType,
)
from xdsl.ir import Dialect, Attribute, Region, SSAValue, Data, Operation
from xdsl.ir.affine import AffineMap, AffineExpr
from xdsl.printer import Printer
from xdsl.parser import Parser
from xdsl.traits import IsTerminator
from xdsl.irdl import (
    irdl_op_definition,
    irdl_attr_definition,
    IRDLOperation,
    VarOperand,
    var_operand_def,
    var_result_def,
    VarOpResult,
    region_def,
    attr_def,
    AttrSizedOperandSegments,
    opt_attr_def,
)
from typing import Sequence
from enum import Enum


class IteratorType(Enum):
    "Iterator type for linalg trait"

    PARALLEL = "parallel"
    REDUCTION = "reduction"
    WINDOW = "window"


@irdl_attr_definition
class IteratorTypeAttr(Data[IteratorType]):
    name = "linalg.iterator_type"

    @staticmethod
    def parse_parameter(parser: Parser) -> IteratorType:
        if parser.parse_optional_keyword("parallel") is not None:
            return IteratorType.PARALLEL
        if parser.parse_optional_keyword("reduction") is not None:
            return IteratorType.REDUCTION
        if parser.parse_optional_keyword("window") is not None:
            return IteratorType.WINDOW
        parser.raise_error("`parallel`, `reduction` or `window` expected")

    def print_parameter(self, printer: Printer) -> None:
        data = self.data
        match data:
            case IteratorType.PARALLEL:
                printer.print_string("parallel")
            case IteratorType.REDUCTION:
                printer.print_string("reduction")
            case IteratorType.WINDOW:
                printer.print_string("window")


@irdl_op_definition
class Generic(IRDLOperation):
    name = "linalg.generic"

    inputs: VarOperand = var_operand_def()
    outputs: VarOperand = var_operand_def(AnyShapedType())

    res: VarOpResult = var_result_def(AnyTensorType)

    body: Region = region_def("single_block")

    # Trait attributes
    indexing_maps: ArrayAttr[AffineMapAttr] = attr_def(ArrayAttr[AffineMapAttr])
    iterator_types: ArrayAttr[IteratorTypeAttr] = attr_def(ArrayAttr[IteratorTypeAttr])
    doc: StringAttr | None = opt_attr_def(StringAttr)
    library_call: StringAttr | None = opt_attr_def(StringAttr)

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        body: Region,
        indexing_maps: Sequence[AffineMapAttr],
        iterator_types: Sequence[Attribute],
        doc: StringAttr | None = None,
        library_call: StringAttr | None = None,
    ) -> None:
        super().__init__(
            operands=[inputs, outputs],
            result_types=[[]],
            attributes={
                "indexing_maps": ArrayAttr(indexing_maps),
                "iterator_types": ArrayAttr(iterator_types),
                "doc": doc,
                "library_call": library_call,
            },
            regions=[body],
        )

    def get_num_loops(self) -> int:
        return self.indexing_maps.data[0].data.num_dims

    def get_loops_to_shapes_map(self) -> AffineMap:
        result_exprs: list[AffineExpr] = []
        for map_attr in self.indexing_maps:
            map = map_attr.data
            result_exprs.extend(map.results)

        dims = self.get_num_loops()
        # FIXME: Check for symbols also. Currently, we do not allow dynamic
        # shapes so this is not possible.
        syms = 0

        return AffineMap(dims, syms, result_exprs)

    def get_shapes_to_loops_map(self) -> AffineMap:
        loops_to_shapes = self.get_loops_to_shapes_map()
        inverse = loops_to_shapes.inverse_permutation()
        if not inverse:
            raise NotImplementedError("Non-invertible map")
        return inverse

    def get_static_shapes(self) -> list[int]:
        sizes: list[int] = []
        for input in self.inputs:
            if isinstance(input.typ, ShapedType):
                for shape in input.typ.get_shape():
                    sizes.append(shape)
        for output in self.outputs:
            if isinstance(output.typ, ShapedType):
                for shape in output.typ.get_shape():
                    sizes.append(shape)
        return sizes

    def get_static_loop_ranges(self):
        shapes_to_loops = self.get_shapes_to_loops_map()
        return shapes_to_loops.eval(self.get_static_shapes(), [])


@irdl_op_definition
class Yield(IRDLOperation):
    name = "linalg.yield"

    values: VarOperand = var_operand_def()

    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation) -> None:
        super().__init__(operands=[SSAValue.get(operand) for operand in operands])


Linalg = Dialect([Generic, Yield], [IteratorTypeAttr])
