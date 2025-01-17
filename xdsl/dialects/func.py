from __future__ import annotations
import re
from typing import Annotated, Union, Sequence, cast

from xdsl.dialects.builtin import (
    StringAttr,
    FunctionType,
    SymbolRefAttr,
)
from xdsl.ir import (
    SSAValue,
    Operation,
    Block,
    Region,
    Attribute,
    Dialect,
    BlockArgument,
)
from xdsl.irdl import (
    VarOpResult,
    irdl_op_definition,
    VarOperand,
    AnyAttr,
    OpAttr,
    OptOpAttr,
    IRDLOperation,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import HasParent
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.lexer import Span


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "func.func"

    body: Region
    sym_name: OpAttr[StringAttr]
    function_type: OpAttr[FunctionType]
    sym_visibility: OptOpAttr[StringAttr]

    def verify_(self) -> None:
        # If this is an empty region (external function), then return
        if len(self.body.blocks) == 0:
            return

        # TODO: how to verify that there is a terminator?
        entry_block: Block = self.body.blocks[0]
        block_arg_types = [arg.typ for arg in entry_block.args]
        if self.function_type.inputs.data != tuple(block_arg_types):
            raise VerifyException(
                "Expected entry block arguments to have the same types as the function "
                "input types"
            )

    @classmethod
    def parse(cls, parser: Parser) -> FuncOp:
        # Parse visibility keyword if present
        visibility = parser.tokenizer.next_token_of_pattern(
            re.compile("public|nested|private")
        )
        if isinstance(visibility, Span):
            visibility = visibility.text

        # Parse function name
        name = parser.parse_symbol_name().data

        def parse_fun_input():
            ret = parser.parse_optional_argument()
            if ret is None:
                ret = parser.parse_optional_type()
            if ret is None:
                parser.raise_error("Expected argument or type")
            return ret

        # Parse function arguments
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN,
            parse_fun_input,
        )

        # Check consistency (They should be either all named or none)
        if isa(args, list[parser.Argument]):
            entry_args = args
            input_types = cast(list[Attribute], [a.type for a in args])
        elif isa(args, list[Attribute]):
            entry_args = None
            input_types = args
        else:
            parser.raise_error(
                "Expected all arguments to be named or all arguments to be unnamed."
            )

        # Parse return type
        if parser.parse_optional_punctuation("->"):
            return_types = parser.parse_optional_type()
            if return_types:
                return_types = [return_types]
            else:
                return_types = parser.parse_comma_separated_list(
                    parser.Delimiter.PAREN, parser.parse_type
                )
        else:
            return_types = []

        attr_dict = parser.parse_optional_attr_dict_with_keyword(
            ("sym_name", "function_type", "sym_visibility")
        )

        # Parse body
        region = parser.parse_optional_region(entry_args)
        if region is None:
            region = Region()
        func = FuncOp.from_region(name, input_types, return_types, region, visibility)
        if attr_dict is not None:
            func.attributes |= attr_dict.data
        return func

    def print(self, printer: Printer):
        if self.sym_visibility:
            visibility = self.sym_visibility.data
            printer.print(f" {visibility}")

        printer.print(f" @{self.sym_name.data}")
        if len(self.body.blocks) > 0:
            printer.print("(")
            printer.print_list(self.body.blocks[0].args, printer.print_block_argument)
            printer.print(") ")
            if self.function_type.outputs:
                printer.print("-> ")
                if len(self.function_type.outputs) > 1:
                    printer.print("(")
                printer.print_list(self.function_type.outputs, printer.print_attribute)
                if len(self.function_type.outputs) > 1:
                    printer.print(")")
                printer.print(" ")
        else:
            printer.print_attribute(self.function_type)
        attr_dict = {
            k: v
            for k, v in self.attributes.items()
            if k not in ("sym_name", "function_type", "sym_visibility")
        }
        if len(attr_dict) > 0:
            printer.print(" attributes {")
            printer.print_list(
                attr_dict.items(), lambda i: printer.print(f'"{i[0]}" = {i[1]}')
            )
            printer.print("}")

        if len(self.body.blocks) > 0:
            printer.print_region(self.body, False, False)

    @staticmethod
    def from_callable(
        name: str,
        input_types: Sequence[Attribute],
        return_types: Sequence[Attribute],
        func: Block.BlockCallback,
    ) -> FuncOp:
        type_attr = FunctionType.from_lists(input_types, return_types)
        attributes: dict[str, Attribute] = {
            "sym_name": StringAttr(name),
            "function_type": type_attr,
            "sym_visibility": StringAttr("private"),
        }
        op = FuncOp.build(
            attributes=attributes,
            regions=[Region(Block.from_callable(input_types, func))],
        )
        return op

    @staticmethod
    def external(
        name: str, input_types: Sequence[Attribute], return_types: Sequence[Attribute]
    ) -> FuncOp:
        type_attr = FunctionType.from_lists(input_types, return_types)
        attributes: dict[str, Attribute] = {
            "sym_name": StringAttr(name),
            "function_type": type_attr,
            "sym_visibility": StringAttr("private"),
        }
        op = FuncOp.build(attributes=attributes, regions=[Region()])
        return op

    @staticmethod
    def from_region(
        name: str,
        input_types: Sequence[Attribute],
        return_types: Sequence[Attribute],
        region: Region,
        visibility: StringAttr | str | None = None,
    ) -> FuncOp:
        if isinstance(visibility, str):
            visibility = StringAttr(visibility)
        type_attr = FunctionType.from_lists(input_types, return_types)
        attributes: dict[str, Attribute | None] = {
            "sym_name": StringAttr(name),
            "function_type": type_attr,
            "sym_visibility": visibility,
        }
        op = FuncOp.build(attributes=attributes, regions=[region])
        return op

    def replace_argument_type(self, arg: int | BlockArgument, new_type: Attribute):
        """
        Replaces the type of the argument specified by arg (either the index of the arg,
        or the BlockArgument object itself) with new_type. This also takes care of updating
        the function_type attribute.
        """
        if isinstance(arg, int):
            try:
                arg = self.body.blocks[0].args[arg]
            except IndexError:
                raise IndexError(
                    "Block {} does not have argument #{}".format(
                        self.body.blocks[0], arg
                    )
                )

        if arg not in self.args:
            raise ValueError("Arg {} does not belong to this function".format(arg))

        arg.typ = new_type
        self.update_function_type()

    def update_function_type(self):
        """
        Update the function_type attribute to reflect changes in the
        block argument types or return statement arguments.
        """
        # Refuse to work with external function definitions, as they don't have block args
        assert (
            not self.is_declaration
        ), "update_function_type does not work with function declarations!"
        return_op = self.get_return_op()
        return_type: tuple[Attribute] = self.function_type.outputs.data

        if return_op is not None:
            return_type = tuple(arg.typ for arg in return_op.operands)

        self.attributes["function_type"] = FunctionType.from_lists(
            [arg.typ for arg in self.args],
            return_type,
        )

    def get_return_op(self) -> Return | None:
        """
        Helper for easily retrieving the return operation of a given
        function. Returns None if it couldn't find a return op.
        """
        if self.is_declaration:
            return None
        ret_op = self.body.blocks[-1].last_op
        if not isinstance(ret_op, Return):
            return None
        return ret_op

    @property
    def args(self) -> tuple[BlockArgument, ...]:
        """
        A helper to quickly get access to the block arguments of the function
        """
        assert (
            not self.is_declaration
        ), "Function declarations don't have BlockArguments!"
        return self.body.blocks[0].args

    @property
    def is_declaration(self) -> bool:
        """
        A helper to identify functions that are external declarations (have an empty
        function body)
        """
        return len(self.body.blocks) == 0


@irdl_op_definition
class Call(IRDLOperation):
    name = "func.call"
    arguments: Annotated[VarOperand, AnyAttr()]
    callee: OpAttr[SymbolRefAttr]

    # Note: naming this results triggers an ArgumentError
    res: Annotated[VarOpResult, AnyAttr()]
    # TODO how do we verify that the types are correct?

    @staticmethod
    def get(
        callee: Union[str, SymbolRefAttr],
        arguments: Sequence[Union[SSAValue, Operation]],
        return_types: Sequence[Attribute],
    ) -> Call:
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)
        return Call.build(
            operands=[arguments],
            result_types=[return_types],
            attributes={"callee": callee},
        )


@irdl_op_definition
class Return(IRDLOperation):
    name = "func.return"
    arguments: Annotated[VarOperand, AnyAttr()]

    traits = frozenset([HasParent(FuncOp)])

    def verify_(self) -> None:
        func_op = self.parent_op()
        assert isinstance(func_op, FuncOp)

        function_return_types = func_op.function_type.outputs.data
        return_types = tuple(arg.typ for arg in self.arguments)
        if function_return_types != return_types:
            raise VerifyException(
                "Expected arguments to have the same types as the function output types"
            )

    @staticmethod
    def get(*ops: Operation | SSAValue) -> Return:
        return Return.build(operands=[list(ops)])


Func = Dialect([FuncOp, Call, Return], [])
