from asyncio import ensure_future, gather
from collections.abc import Mapping
from inspect import isawaitable
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Union,
    Tuple,
    Type,
    cast,
)

from ..error import GraphQLError, located_error
from ..language import (
    DocumentNode,
    FieldNode,
    FragmentDefinitionNode,
    FragmentSpreadNode,
    InlineFragmentNode,
    OperationDefinitionNode,
    OperationType,
    SelectionSetNode,
)
from ..pyutils import (
    inspect,
    is_awaitable as default_is_awaitable,
    AwaitableOrValue,
    FrozenList,
    Path,
    Undefined,
)
from ..utilities.get_operation_root_type import get_operation_root_type
from ..utilities.type_from_ast import type_from_ast
from ..type import (
    GraphQLAbstractType,
    GraphQLField,
    GraphQLIncludeDirective,
    GraphQLLeafType,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLOutputType,
    GraphQLSchema,
    GraphQLSkipDirective,
    GraphQLFieldResolver,
    GraphQLTypeResolver,
    GraphQLResolveInfo,
    SchemaMetaFieldDef,
    TypeMetaFieldDef,
    TypeNameMetaFieldDef,
    assert_valid_schema,
    is_abstract_type,
    is_leaf_type,
    is_list_type,
    is_non_null_type,
    is_object_type,
)
from .middleware import MiddlewareManager
from .values import get_argument_values, get_directive_values, get_variable_values

__all__ = [
    "assert_valid_execution_arguments",
    "default_field_resolver",
    "default_type_resolver",
    "execute",
    "execute_sync",
    "get_field_def",
    "ExecutionResult",
    "ExecutionContext",
    "Middleware",
]


# Terminology
#
# "Definitions" are the generic name for top-level statements in the document.
# Examples of this include:
# 1) Operations (such as a query)
# 2) Fragments
#
# "Operations" are a generic name for requests in the document.
# Examples of this include:
# 1) query,
# 2) mutation
#
# "Selections" are the definitions that can appear legally and at
# single level of the query. These include:
# 1) field references e.g "a"
# 2) fragment "spreads" e.g. "...c"
# 3) inline fragment "spreads" e.g. "...on Type { a }"


class ExecutionResult:
    """The result of GraphQL execution.

    - ``data`` is the result of a successful execution of the query.
    - ``errors`` is included when any errors occurred as a non-empty list.
    - ``extensions`` is reserved for adding non-standard properties.
    """

    __slots__ = "data", "errors", "extensions"

    data: Optional[Dict[str, Any]]
    errors: Optional[List[GraphQLError]]
    extensions: Optional[Dict[str, Any]]

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        errors: Optional[List[GraphQLError]] = None,
        extensions: Optional[Dict[str, Any]] = None,
    ):
        self.data = data
        self.errors = errors
        self.extensions = extensions

    def __repr__(self) -> str:
        name = self.__class__.__name__
        ext = "" if self.extensions is None else f", extensions={self.extensions}"
        return f"{name}(data={self.data!r}, errors={self.errors!r}{ext})"

    def __iter__(self) -> Iterable[Any]:
        return iter((self.data, self.errors))

    @property
    def formatted(self) -> Dict[str, Any]:
        """Get execution result formatted according to the specification."""
        if self.extensions is None:
            return dict(data=self.data, errors=self.errors)
        return dict(data=self.data, errors=self.errors, extensions=self.extensions)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, dict):
            if "extensions" not in other:
                return other == dict(data=self.data, errors=self.errors)
            return other == dict(
                data=self.data, errors=self.errors, extensions=self.extensions
            )
        if isinstance(other, tuple):
            if len(other) == 2:
                return other == (self.data, self.errors)
            return other == (self.data, self.errors, self.extensions)
        return (
            isinstance(other, self.__class__)
            and other.data == self.data
            and other.errors == self.errors
            and other.extensions == self.extensions
        )

    def __ne__(self, other: Any) -> bool:
        return not self == other


Middleware = Optional[Union[Tuple, List, MiddlewareManager]]


class ExecutionContext:
    """Data that must be available at all points during query execution.

    Namely, schema of the type system that is currently executing, and the fragments
    defined in the query document.
    """

    schema: GraphQLSchema
    fragments: Dict[str, FragmentDefinitionNode]
    root_value: Any
    context_value: Any
    operation: OperationDefinitionNode
    variable_values: Dict[str, Any]
    field_resolver: GraphQLFieldResolver
    type_resolver: GraphQLTypeResolver
    errors: List[GraphQLError]
    middleware_manager: Optional[MiddlewareManager]

    is_awaitable = staticmethod(default_is_awaitable)

    def __init__(
        self,
        schema: GraphQLSchema,
        fragments: Dict[str, FragmentDefinitionNode],
        root_value: Any,
        context_value: Any,
        operation: OperationDefinitionNode,
        variable_values: Dict[str, Any],
        field_resolver: GraphQLFieldResolver,
        type_resolver: GraphQLTypeResolver,
        errors: List[GraphQLError],
        middleware_manager: Optional[MiddlewareManager],
        is_awaitable: Optional[Callable[[Any], bool]],
    ) -> None:
        self.schema = schema
        self.fragments = fragments
        self.root_value = root_value
        self.context_value = context_value
        self.operation = operation
        self.variable_values = variable_values
        self.field_resolver = field_resolver  # type: ignore
        self.type_resolver = type_resolver  # type: ignore
        self.errors = errors
        self.middleware_manager = middleware_manager
        if is_awaitable:
            self.is_awaitable = is_awaitable
        self._subfields_cache: Dict[Tuple, Dict[str, List[FieldNode]]] = {}

    @classmethod
    def build(
        cls,
        schema: GraphQLSchema,
        document: DocumentNode,
        root_value: Any = None,
        context_value: Any = None,
        raw_variable_values: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        field_resolver: Optional[GraphQLFieldResolver] = None,
        type_resolver: Optional[GraphQLTypeResolver] = None,
        middleware: Optional[Middleware] = None,
        is_awaitable: Optional[Callable[[Any], bool]] = None,
    ) -> Union[List[GraphQLError], "ExecutionContext"]:
        """Build an execution context

        Constructs a ExecutionContext object from the arguments passed to execute, which
        we will pass throughout the other execution methods.

        Throws a GraphQLError if a valid execution context cannot be created.

        For internal use only.
        """
        operation: Optional[OperationDefinitionNode] = None
        fragments: Dict[str, FragmentDefinitionNode] = {}
        middleware_manager: Optional[MiddlewareManager] = None
        if middleware is not None:
            if isinstance(middleware, (list, tuple)):
                middleware_manager = MiddlewareManager(*middleware)
            elif isinstance(middleware, MiddlewareManager):
                middleware_manager = middleware
            else:
                raise TypeError(
                    "Middleware must be passed as a list or tuple of functions"
                    " or objects, or as a single MiddlewareManager object."
                    f" Got {inspect(middleware)} instead."
                )

        for definition in document.definitions:
            if isinstance(definition, OperationDefinitionNode):
                if operation_name is None:
                    if operation:
                        return [
                            GraphQLError(
                                "Must provide operation name"
                                " if query contains multiple operations."
                            )
                        ]
                    operation = definition
                elif definition.name and definition.name.value == operation_name:
                    operation = definition
            elif isinstance(definition, FragmentDefinitionNode):
                fragments[definition.name.value] = definition

        if not operation:
            if operation_name is not None:
                return [GraphQLError(f"Unknown operation named '{operation_name}'.")]
            return [GraphQLError("Must provide an operation.")]

        coerced_variable_values = get_variable_values(
            schema,
            operation.variable_definitions or FrozenList(),
            raw_variable_values or {},
            max_errors=50,
        )

        if isinstance(coerced_variable_values, list):
            return coerced_variable_values  # errors

        return cls(
            schema,
            fragments,
            root_value,
            context_value,
            operation,
            coerced_variable_values,  # coerced values
            field_resolver or default_field_resolver,
            type_resolver or default_type_resolver,
            [],
            middleware_manager,
            is_awaitable,
        )

    def build_response(
        self, data: AwaitableOrValue[Optional[Dict[str, Any]]]
    ) -> AwaitableOrValue[ExecutionResult]:
        """Build response.

        Given a completed execution context and data, build the (data, errors) response
        defined by the "Response" section of the GraphQL spec.
        """
        if self.is_awaitable(data):

            async def build_response_async() -> ExecutionResult:
                return self.build_response(await data)  # type: ignore

            return build_response_async()
        data = cast(Optional[Dict[str, Any]], data)
        errors = self.errors
        if not errors:
            return ExecutionResult(data, None)
        # Sort the error list in order to make it deterministic, since we might have
        # been using parallel execution.
        errors.sort(
            key=lambda error: (error.locations or [], error.path or [], error.message)
        )
        return ExecutionResult(data, errors)

    def execute_operation(
        self, operation: OperationDefinitionNode, root_value: Any
    ) -> Optional[AwaitableOrValue[Any]]:
        """Execute an operation.

        Implements the "Evaluating operations" section of the spec.
        """
        type_ = get_operation_root_type(self.schema, operation)
        fields = self.collect_fields(type_, operation.selection_set, {}, set())

        path = None

        # Errors from sub-fields of a NonNull type may propagate to the top level, at
        # which point we still log the error and null the parent field, which in this
        # case is the entire response.
        #
        # Similar to complete_value_catching_error.
        try:
            # noinspection PyArgumentList
            result = (
                self.execute_fields_serially
                if operation.operation == OperationType.MUTATION
                else self.execute_fields
            )(type_, root_value, path, fields)
        except GraphQLError as error:
            self.errors.append(error)
            return None
        else:
            if self.is_awaitable(result):
                # noinspection PyShadowingNames
                async def await_result() -> Any:
                    try:
                        return await result  # type: ignore
                    except GraphQLError as error:
                        self.errors.append(error)

                return await_result()
            return result

    def execute_fields_serially(
        self,
        parent_type: GraphQLObjectType,
        source_value: Any,
        path: Optional[Path],
        fields: Dict[str, List[FieldNode]],
    ) -> AwaitableOrValue[Dict[str, Any]]:
        """Execute the given fields serially.

        Implements the "Evaluating selection sets" section of the spec for "write" mode.
        """
        results: AwaitableOrValue[Dict[str, Any]] = {}
        is_awaitable = self.is_awaitable
        for response_name, field_nodes in fields.items():
            field_path = Path(path, response_name, parent_type.name)
            result = self.resolve_field(
                parent_type, source_value, field_nodes, field_path
            )
            if result is Undefined:
                continue
            if is_awaitable(results):
                # noinspection PyShadowingNames
                async def await_and_set_result(
                    results: Awaitable[Dict[str, Any]],
                    response_name: str,
                    result: AwaitableOrValue[Any],
                ) -> Dict[str, Any]:
                    awaited_results = await results
                    awaited_results[response_name] = (
                        await result if is_awaitable(result) else result
                    )
                    return awaited_results

                results = await_and_set_result(
                    cast(Awaitable, results), response_name, result
                )
            elif is_awaitable(result):
                # noinspection PyShadowingNames
                async def set_result(
                    results: Dict[str, Any],
                    response_name: str,
                    result: Awaitable,
                ) -> Dict[str, Any]:
                    results[response_name] = await result
                    return results

                results = set_result(
                    cast(Dict[str, Any], results), response_name, result
                )
            else:
                cast(Dict[str, Any], results)[response_name] = result
        if is_awaitable(results):
            # noinspection PyShadowingNames
            async def get_results() -> Any:
                return await cast(Awaitable, results)

            return get_results()
        return results

    def execute_fields(
        self,
        parent_type: GraphQLObjectType,
        source_value: Any,
        path: Optional[Path],
        fields: Dict[str, List[FieldNode]],
    ) -> AwaitableOrValue[Dict[str, Any]]:
        """Execute the given fields concurrently.

        Implements the "Evaluating selection sets" section of the spec for "read" mode.
        """
        results = {}
        is_awaitable = self.is_awaitable
        awaitable_fields: List[str] = []
        append_awaitable = awaitable_fields.append
        for response_name, field_nodes in fields.items():
            field_path = Path(path, response_name, parent_type.name)
            result = self.resolve_field(
                parent_type, source_value, field_nodes, field_path
            )
            if result is not Undefined:
                results[response_name] = result
                if is_awaitable(result):
                    append_awaitable(response_name)

        #  If there are no coroutines, we can just return the object
        if not awaitable_fields:
            return results

        # Otherwise, results is a map from field name to the result of resolving that
        # field, which is possibly a coroutine object. Return a coroutine object that
        # will yield this same map, but with any coroutines awaited in parallel and
        # replaced with the values they yielded.
        async def get_results() -> Dict[str, Any]:
            results.update(
                zip(
                    awaitable_fields,
                    await gather(*(results[field] for field in awaitable_fields)),
                )
            )
            return results

        return get_results()

    def collect_fields(
        self,
        runtime_type: GraphQLObjectType,
        selection_set: SelectionSetNode,
        fields: Dict[str, List[FieldNode]],
        visited_fragment_names: Set[str],
    ) -> Dict[str, List[FieldNode]]:
        """Collect fields.

        Given a selection_set, adds all of the fields in that selection to the passed in
        map of fields, and returns it at the end.

        collect_fields requires the "runtime type" of an object. For a field which
        returns an Interface or Union type, the "runtime type" will be the actual
        Object type returned by that field.

        For internal use only.
        """
        for selection in selection_set.selections:
            if isinstance(selection, FieldNode):
                if not self.should_include_node(selection):
                    continue
                name = get_field_entry_key(selection)
                fields.setdefault(name, []).append(selection)
            elif isinstance(selection, InlineFragmentNode):
                if not self.should_include_node(
                    selection
                ) or not self.does_fragment_condition_match(selection, runtime_type):
                    continue
                self.collect_fields(
                    runtime_type,
                    selection.selection_set,
                    fields,
                    visited_fragment_names,
                )
            elif isinstance(selection, FragmentSpreadNode):  # pragma: no cover else
                frag_name = selection.name.value
                if frag_name in visited_fragment_names or not self.should_include_node(
                    selection
                ):
                    continue
                visited_fragment_names.add(frag_name)
                fragment = self.fragments.get(frag_name)
                if not fragment or not self.does_fragment_condition_match(
                    fragment, runtime_type
                ):
                    continue
                self.collect_fields(
                    runtime_type, fragment.selection_set, fields, visited_fragment_names
                )
        return fields

    def should_include_node(
        self, node: Union[FragmentSpreadNode, FieldNode, InlineFragmentNode]
    ) -> bool:
        """Check if node should be included

        Determines if a field should be included based on the @include and @skip
        directives, where @skip has higher precedence than @include.
        """
        skip = get_directive_values(GraphQLSkipDirective, node, self.variable_values)
        if skip and skip["if"]:
            return False

        include = get_directive_values(
            GraphQLIncludeDirective, node, self.variable_values
        )
        if include and not include["if"]:
            return False

        return True

    def does_fragment_condition_match(
        self,
        fragment: Union[FragmentDefinitionNode, InlineFragmentNode],
        type_: GraphQLObjectType,
    ) -> bool:
        """Determine if a fragment is applicable to the given type."""
        type_condition_node = fragment.type_condition
        if not type_condition_node:
            return True
        conditional_type = type_from_ast(self.schema, type_condition_node)
        if conditional_type is type_:
            return True
        if is_abstract_type(conditional_type):
            return self.schema.is_sub_type(
                cast(GraphQLAbstractType, conditional_type), type_
            )
        return False

    def build_resolve_info(
        self,
        field_def: GraphQLField,
        field_nodes: List[FieldNode],
        parent_type: GraphQLObjectType,
        path: Path,
    ) -> GraphQLResolveInfo:
        """Build the GraphQLResolveInfo object.

        For internal use only."""
        # The resolve function's first argument is a collection of information about
        # the current execution state.
        return GraphQLResolveInfo(
            field_nodes[0].name.value,
            field_nodes,
            field_def.type,
            parent_type,
            path,
            self.schema,
            self.fragments,
            self.root_value,
            self.operation,
            self.variable_values,
            self.context_value,
            self.is_awaitable,
        )

    def resolve_field(
        self,
        parent_type: GraphQLObjectType,
        source: Any,
        field_nodes: List[FieldNode],
        path: Path,
    ) -> AwaitableOrValue[Any]:
        """Resolve the field on the given source object.

        In particular, this figures out the value that the field returns by calling its
        resolve function, then calls complete_value to await coroutine objects,
        serialize scalars, or execute the sub-selection-set for objects.
        """
        field_node = field_nodes[0]
        field_name = field_node.name.value

        field_def = get_field_def(self.schema, parent_type, field_name)
        if not field_def:
            return Undefined

        resolve_fn = field_def.resolve or self.field_resolver

        if self.middleware_manager:
            resolve_fn = self.middleware_manager.get_field_resolver(resolve_fn)

        info = self.build_resolve_info(field_def, field_nodes, parent_type, path)

        # Get the resolve function, regardless of if its result is normal or abrupt
        # (error).
        result = self.resolve_field_value_or_error(
            field_def, field_nodes, resolve_fn, source, info
        )

        return self.complete_value_catching_error(
            field_def.type, field_nodes, info, path, result
        )

    def resolve_field_value_or_error(
        self,
        field_def: GraphQLField,
        field_nodes: List[FieldNode],
        resolve_fn: GraphQLFieldResolver,
        source: Any,
        info: GraphQLResolveInfo,
    ) -> Union[Exception, Any]:
        """Resolve field to a value or an error.

        Isolates the "ReturnOrAbrupt" behavior to not de-opt the resolve_field()
        method. Returns the result of resolveFn or the abrupt-return Error object.

        For internal use only.
        """
        try:
            # Build a dictionary of arguments from the field.arguments AST, using the
            # variables scope to fulfill any variable references.
            args = get_argument_values(field_def, field_nodes[0], self.variable_values)

            # Note that contrary to the JavaScript implementation, we pass the context
            # value as part of the resolve info.
            result = resolve_fn(source, info, **args)
            if self.is_awaitable(result):
                # noinspection PyShadowingNames
                async def await_result() -> Any:
                    try:
                        intermediate_result = await result
                        if isawaitable(intermediate_result):
                            return await intermediate_result
                        return intermediate_result
                    except Exception as error:
                        return error

                return await_result()
            return result
        except Exception as error:
            return error

    def complete_value_catching_error(
        self,
        return_type: GraphQLOutputType,
        field_nodes: List[FieldNode],
        info: GraphQLResolveInfo,
        path: Path,
        result: Any,
    ) -> AwaitableOrValue[Any]:
        """Complete a value while catching an error.

        This is a small wrapper around completeValue which detects and logs errors in
        the execution context.
        """
        completed: AwaitableOrValue[Any]
        try:
            if self.is_awaitable(result):

                async def await_result() -> Any:
                    value = self.complete_value(
                        return_type, field_nodes, info, path, await result
                    )
                    if self.is_awaitable(value):
                        return await value
                    return value

                completed = await_result()
            else:
                completed = self.complete_value(
                    return_type, field_nodes, info, path, result
                )
            if self.is_awaitable(completed):
                # noinspection PyShadowingNames
                async def await_completed() -> Any:
                    try:
                        return await completed
                    except Exception as error:
                        self.handle_field_error(error, field_nodes, path, return_type)

                return await_completed()
            return completed
        except Exception as error:
            self.handle_field_error(error, field_nodes, path, return_type)
            return None

    def handle_field_error(
        self,
        raw_error: Exception,
        field_nodes: List[FieldNode],
        path: Path,
        return_type: GraphQLOutputType,
    ) -> None:
        error = located_error(raw_error, field_nodes, path.as_list())

        # If the field type is non-nullable, then it is resolved without any protection
        # from errors, however it still properly locates the error.
        if is_non_null_type(return_type):
            raise error
        # Otherwise, error protection is applied, logging the error and resolving a
        # null value for this field if one is encountered.
        self.errors.append(error)
        return None

    def complete_value(
        self,
        return_type: GraphQLOutputType,
        field_nodes: List[FieldNode],
        info: GraphQLResolveInfo,
        path: Path,
        result: Any,
    ) -> AwaitableOrValue[Any]:
        """Complete a value.

        Implements the instructions for completeValue as defined in the "Field entries"
        section of the spec.

        If the field type is Non-Null, then this recursively completes the value for the
        inner type. It throws a field error if that completion returns null, as per the
        "Nullability" section of the spec.

        If the field type is a List, then this recursively completes the value for the
        inner type on each item in the list.

        If the field type is a Scalar or Enum, ensures the completed value is a legal
        value of the type by calling the ``serialize`` method of GraphQL type
        definition.

        If the field is an abstract type, determine the runtime type of the value and
        then complete based on that type.

        Otherwise, the field type expects a sub-selection set, and will complete the
        value by evaluating all sub-selections.
        """
        # If result is an Exception, throw a located error.
        if isinstance(result, Exception):
            raise result

        # If field type is NonNull, complete for inner type, and throw field error if
        # result is null.
        if is_non_null_type(return_type):
            completed = self.complete_value(
                cast(GraphQLNonNull, return_type).of_type,
                field_nodes,
                info,
                path,
                result,
            )
            if completed is None:
                raise TypeError(
                    "Cannot return null for non-nullable field"
                    f" {info.parent_type.name}.{info.field_name}."
                )
            return completed

        # If result value is null or undefined then return null.
        if result is None or result is Undefined:
            return None

        # If field type is List, complete each item in the list with inner type
        if is_list_type(return_type):
            return self.complete_list_value(
                cast(GraphQLList, return_type), field_nodes, info, path, result
            )

        # If field type is a leaf type, Scalar or Enum, serialize to a valid value,
        # returning null if serialization is not possible.
        if is_leaf_type(return_type):
            return self.complete_leaf_value(cast(GraphQLLeafType, return_type), result)

        # If field type is an abstract type, Interface or Union, determine the runtime
        # Object type and complete for that type.
        if is_abstract_type(return_type):
            return self.complete_abstract_value(
                cast(GraphQLAbstractType, return_type), field_nodes, info, path, result
            )

        # If field type is Object, execute and complete all sub-selections.
        if is_object_type(return_type):
            return self.complete_object_value(
                cast(GraphQLObjectType, return_type), field_nodes, info, path, result
            )

        # Not reachable. All possible output types have been considered.
        raise TypeError(  # pragma: no cover
            "Cannot complete value of unexpected output type:"
            f" '{inspect(return_type)}'."
        )

    def complete_list_value(
        self,
        return_type: GraphQLList[GraphQLOutputType],
        field_nodes: List[FieldNode],
        info: GraphQLResolveInfo,
        path: Path,
        result: Iterable[Any],
    ) -> AwaitableOrValue[Any]:
        """Complete a list value.

        Complete a list value by completing each item in the list with the inner type.
        """
        if not isinstance(result, Iterable) or isinstance(result, str):
            raise GraphQLError(
                "Expected Iterable, but did not find one for field"
                f" '{info.parent_type.name}.{info.field_name}'."
            )

        # This is specified as a simple map, however we're optimizing the path where
        # the list contains no coroutine objects by avoiding creating another coroutine
        # object.
        item_type = return_type.of_type
        is_awaitable = self.is_awaitable
        awaitable_indices: List[int] = []
        append_awaitable = awaitable_indices.append
        completed_results: List[Any] = []
        append_result = completed_results.append
        for index, item in enumerate(result):
            # No need to modify the info object containing the path, since from here on
            # it is not ever accessed by resolver functions.
            field_path = path.add_key(index, None)
            completed_item = self.complete_value_catching_error(
                item_type, field_nodes, info, field_path, item
            )

            if is_awaitable(completed_item):
                append_awaitable(index)
            append_result(completed_item)

        if not awaitable_indices:
            return completed_results

        # noinspection PyShadowingNames
        async def get_completed_results() -> Any:
            for index, result in zip(
                awaitable_indices,
                await gather(
                    *(completed_results[index] for index in awaitable_indices)
                ),
            ):
                completed_results[index] = result
            return completed_results

        return get_completed_results()

    @staticmethod
    def complete_leaf_value(return_type: GraphQLLeafType, result: Any) -> Any:
        """Complete a leaf value.

        Complete a Scalar or Enum by serializing to a valid value, returning null if
        serialization is not possible.
        """
        serialized_result = return_type.serialize(result)
        if serialized_result is Undefined:
            raise TypeError(
                f"Expected a value of type '{inspect(return_type)}'"
                f" but received: {inspect(result)}"
            )
        return serialized_result

    def complete_abstract_value(
        self,
        return_type: GraphQLAbstractType,
        field_nodes: List[FieldNode],
        info: GraphQLResolveInfo,
        path: Path,
        result: Any,
    ) -> AwaitableOrValue[Any]:
        """Complete an abstract value.

        Complete a value of an abstract type by determining the runtime object type of
        that value, then complete the value for that type.
        """
        resolve_type_fn = return_type.resolve_type or self.type_resolver
        runtime_type = resolve_type_fn(result, info, return_type)  # type: ignore

        if self.is_awaitable(runtime_type):

            async def await_complete_object_value() -> Any:
                value = self.complete_object_value(
                    self.ensure_valid_runtime_type(
                        await runtime_type,  # type: ignore
                        return_type,
                        field_nodes,
                        info,
                        result,
                    ),
                    field_nodes,
                    info,
                    path,
                    result,
                )
                if self.is_awaitable(value):
                    return await value  # type: ignore
                return value

            return await_complete_object_value()
        runtime_type = cast(Optional[Union[GraphQLObjectType, str]], runtime_type)

        return self.complete_object_value(
            self.ensure_valid_runtime_type(
                runtime_type, return_type, field_nodes, info, result
            ),
            field_nodes,
            info,
            path,
            result,
        )

    def ensure_valid_runtime_type(
        self,
        runtime_type_or_name: Optional[Union[GraphQLObjectType, str]],
        return_type: GraphQLAbstractType,
        field_nodes: List[FieldNode],
        info: GraphQLResolveInfo,
        result: Any,
    ) -> GraphQLObjectType:
        runtime_type = (
            self.schema.get_type(runtime_type_or_name)
            if isinstance(runtime_type_or_name, str)
            else runtime_type_or_name
        )

        if not is_object_type(runtime_type):
            raise GraphQLError(
                f"Abstract type '{return_type.name}' must resolve"
                " to an Object type at runtime"
                f" for field '{info.parent_type.name}.{info.field_name}'"
                f" with value {inspect(result)}, received '{inspect(runtime_type)}'."
                f" Either the '{return_type.name}' type should provide"
                " a 'resolve_type' function or each possible type should"
                " provide an 'is_type_of' function.",
                field_nodes,
            )
        runtime_type = cast(GraphQLObjectType, runtime_type)

        if not self.schema.is_sub_type(return_type, runtime_type):
            raise GraphQLError(
                f"Runtime Object type '{runtime_type.name}' is not a possible"
                f" type for '{return_type.name}'.",
                field_nodes,
            )

        return runtime_type

    def complete_object_value(
        self,
        return_type: GraphQLObjectType,
        field_nodes: List[FieldNode],
        info: GraphQLResolveInfo,
        path: Path,
        result: Any,
    ) -> AwaitableOrValue[Dict[str, Any]]:
        """Complete an Object value by executing all sub-selections."""
        # If there is an `is_type_of()` predicate function, call it with the current
        # result. If `is_type_of()` returns False, then raise an error rather than
        #  continuing execution.
        if return_type.is_type_of:
            is_type_of = return_type.is_type_of(result, info)

            if self.is_awaitable(is_type_of):

                async def collect_and_execute_subfields_async() -> Dict[str, Any]:
                    if not await is_type_of:  # type: ignore
                        raise invalid_return_type_error(
                            return_type, result, field_nodes
                        )
                    return self.collect_and_execute_subfields(
                        return_type, field_nodes, path, result
                    )  # type: ignore

                return collect_and_execute_subfields_async()

            if not is_type_of:
                raise invalid_return_type_error(return_type, result, field_nodes)

        return self.collect_and_execute_subfields(
            return_type, field_nodes, path, result
        )

    def collect_and_execute_subfields(
        self,
        return_type: GraphQLObjectType,
        field_nodes: List[FieldNode],
        path: Path,
        result: Any,
    ) -> AwaitableOrValue[Dict[str, Any]]:
        """Collect sub-fields to execute to complete this value."""
        sub_field_nodes = self.collect_subfields(return_type, field_nodes)
        return self.execute_fields(return_type, result, path, sub_field_nodes)

    def collect_subfields(
        self, return_type: GraphQLObjectType, field_nodes: List[FieldNode]
    ) -> Dict[str, List[FieldNode]]:
        """Collect subfields.

        A cached collection of relevant subfields with regard to the return type is
        kept in the execution context as ``_subfields_cache``. This ensures the
        subfields are not repeatedly calculated, which saves overhead when resolving
        lists of values.
        """
        cache = self._subfields_cache
        # We cannot use the field_nodes themselves as key for the cache, since they
        # are not hashable as a list. We also do not want to use the field_nodes
        # themselves (converted to a tuple) as keys, since hashing them is slow.
        # Therefore we use the ids of the field_nodes as keys. Note that we do not
        # use the id of the list, since we want to hit the cache for all lists of
        # the same nodes, not only for the same list of nodes. Also, the list id may
        # even be reused, in which case we would get wrong results from the cache.
        key = (
            (return_type, id(field_nodes[0]))
            if len(field_nodes) == 1  # optimize most frequent case
            else tuple((return_type, *map(id, field_nodes)))
        )
        sub_field_nodes = cache.get(key)
        if sub_field_nodes is None:
            sub_field_nodes = {}
            visited_fragment_names: Set[str] = set()
            for field_node in field_nodes:
                selection_set = field_node.selection_set
                if selection_set:
                    sub_field_nodes = self.collect_fields(
                        return_type,
                        selection_set,
                        sub_field_nodes,
                        visited_fragment_names,
                    )
            cache[key] = sub_field_nodes
        return sub_field_nodes


def execute(
    schema: GraphQLSchema,
    document: DocumentNode,
    root_value: Any = None,
    context_value: Any = None,
    variable_values: Optional[Dict[str, Any]] = None,
    operation_name: Optional[str] = None,
    field_resolver: Optional[GraphQLFieldResolver] = None,
    type_resolver: Optional[GraphQLTypeResolver] = None,
    middleware: Optional[Middleware] = None,
    execution_context_class: Optional[Type["ExecutionContext"]] = None,
    is_awaitable: Optional[Callable[[Any], bool]] = None,
) -> AwaitableOrValue[ExecutionResult]:
    """Execute a GraphQL operation.

    Implements the "Evaluating requests" section of the GraphQL specification.

    Returns an ExecutionResult (if all encountered resolvers are synchronous),
    or a coroutine object eventually yielding an ExecutionResult.

    If the arguments to this function do not result in a legal execution context,
    a GraphQLError will be thrown immediately explaining the invalid input.
    """
    # If arguments are missing or incorrect, throw an error.
    assert_valid_execution_arguments(schema, document, variable_values)

    if execution_context_class is None:
        execution_context_class = ExecutionContext

    # If a valid execution context cannot be created due to incorrect arguments,
    # a "Response" with only errors is returned.
    exe_context = execution_context_class.build(
        schema,
        document,
        root_value,
        context_value,
        variable_values,
        operation_name,
        field_resolver,
        type_resolver,
        middleware,
        is_awaitable,
    )

    # Return early errors if execution context failed.
    if isinstance(exe_context, list):
        return ExecutionResult(data=None, errors=exe_context)

    # Return a possible coroutine object that will eventually yield the data described
    # by the "Response" section of the GraphQL specification.
    #
    # If errors are encountered while executing a GraphQL field, only that field and
    # its descendants will be omitted, and sibling fields will still be executed. An
    # execution which encounters errors will still result in a coroutine object that
    # can be executed without errors.

    data = exe_context.execute_operation(exe_context.operation, root_value)
    return exe_context.build_response(data)


def assume_not_awaitable(_value: Any) -> bool:
    """Replacement for isawaitable if everything is assumed to be synchronous."""
    return False


def execute_sync(
    schema: GraphQLSchema,
    document: DocumentNode,
    root_value: Any = None,
    context_value: Any = None,
    variable_values: Optional[Dict[str, Any]] = None,
    operation_name: Optional[str] = None,
    field_resolver: Optional[GraphQLFieldResolver] = None,
    type_resolver: Optional[GraphQLTypeResolver] = None,
    middleware: Optional[Middleware] = None,
    execution_context_class: Optional[Type["ExecutionContext"]] = None,
    check_sync: bool = False,
) -> ExecutionResult:
    """Execute a GraphQL operation synchronously.

    Also implements the "Evaluating requests" section of the GraphQL specification.

    However, it guarantees to complete synchronously (or throw an error) assuming
    that all field resolvers are also synchronous.

    Set check_sync to True to still run checks that no awaitable values are returned.
    """
    is_awaitable = (
        check_sync
        if callable(check_sync)
        else (None if check_sync else assume_not_awaitable)
    )

    result = execute(
        schema,
        document,
        root_value,
        context_value,
        variable_values,
        operation_name,
        field_resolver,
        type_resolver,
        middleware,
        execution_context_class,
        is_awaitable,
    )

    # Assert that the execution was synchronous.
    if isawaitable(result):
        ensure_future(cast(Awaitable[ExecutionResult], result)).cancel()
        raise RuntimeError("GraphQL execution failed to complete synchronously.")

    return cast(ExecutionResult, result)


def assert_valid_execution_arguments(
    schema: GraphQLSchema,
    document: DocumentNode,
    raw_variable_values: Optional[Dict[str, Any]] = None,
) -> None:
    """Check that the arguments are acceptable.

    Essential assertions before executing to provide developer feedback for improper use
    of the GraphQL library.

    For internal use only.
    """
    if not document:
        raise TypeError("Must provide document.")

    # If the schema used for execution is invalid, throw an error.
    assert_valid_schema(schema)

    # Variables, if provided, must be a dictionary.
    if not (raw_variable_values is None or isinstance(raw_variable_values, dict)):
        raise TypeError(
            "Variable values must be provided as a dictionary"
            " with variable names as keys. Perhaps look to see"
            " if an unparsed JSON string was provided."
        )


def get_field_def(
    schema: GraphQLSchema, parent_type: GraphQLObjectType, field_name: str
) -> GraphQLField:
    """Get field definition.

    This method looks up the field on the given type definition. It has special casing
    for the three introspection fields, ``__schema``, ``__type`, and ``__typename``.
    ``__typename`` is special because it can always be queried as a field, even in
    situations where no other fields are allowed, like on a Union. ``__schema`` and
    ``__type`` could get automatically added to the query type, but that would require
    mutating type definitions, which would cause issues.

    For internal use only.
    """
    if field_name == "__schema" and schema.query_type == parent_type:
        return SchemaMetaFieldDef
    elif field_name == "__type" and schema.query_type == parent_type:
        return TypeMetaFieldDef
    elif field_name == "__typename":
        return TypeNameMetaFieldDef
    return parent_type.fields.get(field_name)


def get_field_entry_key(node: FieldNode) -> str:
    """Implements the logic to compute the key of a given field's entry"""
    return node.alias.value if node.alias else node.name.value


def invalid_return_type_error(
    return_type: GraphQLObjectType, result: Any, field_nodes: List[FieldNode]
) -> GraphQLError:
    """Create a GraphQLError for an invalid return type."""
    return GraphQLError(
        f"Expected value of type '{return_type.name}' but got: {inspect(result)}.",
        field_nodes,
    )


def get_typename(value: Any) -> Optional[str]:
    """Get the ``__typename`` property of the given value."""
    if isinstance(value, Mapping):
        return value.get("__typename")
    # need to de-mangle the attribute assumed to be "private" in Python
    for cls in value.__class__.__mro__:
        __typename = getattr(value, f"_{cls.__name__}__typename", None)
        if __typename:
            return __typename
    return None


def default_type_resolver(
    value: Any, info: GraphQLResolveInfo, abstract_type: GraphQLAbstractType
) -> AwaitableOrValue[Optional[Union[GraphQLObjectType, str]]]:
    """Default type resolver function.

    If a resolve_type function is not given, then a default resolve behavior is used
    which attempts two strategies:

    First, See if the provided value has a ``__typename`` field defined, if so, use that
    value as name of the resolved type.

    Otherwise, test each possible type for the abstract type by calling
    :meth:`~graphql.type.GraphQLObjectType.is_type_of` for the object
    being coerced, returning the first type that matches.
    """
    # First, look for `__typename`.
    type_name = get_typename(value)
    if isinstance(type_name, str):
        return type_name

    # Otherwise, test each possible type.
    possible_types = info.schema.get_possible_types(abstract_type)
    is_awaitable = info.is_awaitable
    awaitable_is_type_of_results: List[Awaitable] = []
    append_awaitable_results = awaitable_is_type_of_results.append
    awaitable_types: List[GraphQLObjectType] = []
    append_awaitable_types = awaitable_types.append

    for type_ in possible_types:
        if type_.is_type_of:
            is_type_of_result = type_.is_type_of(value, info)

            if is_awaitable(is_type_of_result):
                append_awaitable_results(cast(Awaitable, is_type_of_result))
                append_awaitable_types(type_)
            elif is_type_of_result:
                return type_

    if awaitable_is_type_of_results:
        # noinspection PyShadowingNames
        async def get_type() -> Optional[Union[GraphQLObjectType, str]]:
            is_type_of_results = await gather(*awaitable_is_type_of_results)
            for is_type_of_result, type_ in zip(is_type_of_results, awaitable_types):
                if is_type_of_result:
                    return type_
            return None

        return get_type()

    return None


def default_field_resolver(source: Any, info: GraphQLResolveInfo, **args: Any) -> Any:
    """Default field resolver.

    If a resolve function is not given, then a default resolve behavior is used which
    takes the property of the source object of the same name as the field and returns
    it as the result, or if it's a function, returns the result of calling that function
    while passing along args and context.

    For dictionaries, the field names are used as keys, for all other objects they are
    used as attribute names.
    """
    # Ensure source is a value for which property access is acceptable.
    field_name = info.field_name
    value = (
        source.get(field_name)
        if isinstance(source, Mapping)
        else getattr(source, field_name, None)
    )
    if callable(value):
        return value(info, **args)
    return value
