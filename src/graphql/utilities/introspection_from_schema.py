import sys

from ..pyutils import FrozenList

if sys.version_info < (3, 7):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict, Optional, Generic, TypeVar

from typing import Any

from ..error import GraphQLError
from ..language import parse, DirectiveLocation
from ..type import GraphQLSchema
from .get_introspection_query import get_introspection_query

__all__ = ["introspection_from_schema"]

# Based on the GraphQL-JS implementation
# https://github.com/graphql/graphql-js/blob/master/src/utilities/getIntrospectionQuery.js

class IntrospectionType:
    __slots__ = "name", "description"

    name: str
    description: Optional[str]

class IntrospectionScalarType(IntrospectionType):
    description: Optional[str]
    specified_by_url: Optional[str]

class IntrospectionObjectType(IntrospectionType):
    kind: str = "OBJECT"
    fields: FrozenList["IntrospectionField"]
    interfaces: FrozenList["IntrospectionInterfaceType"]

class IntrospectionInterfaceType(IntrospectionType):
    kind: str = "INTERFACE"
    fields: FrozenList["IntrospectionField"]
    interfaces: FrozenList["IntrospectionInterfaceType"]
    possible_types: FrozenList["IntrospectionObjectType"]

class IntrospectionUnionType (IntrospectionType):
    kind: str = "UNION"
    possible_types: FrozenList["IntrospectionNamedTypeRef"]

class IntrospectionEnumType(IntrospectionType):
    kind: str = "ENUM"
    enum_values: FrozenList["IntrospectionEnumValue"]

class IntrospectionInputObjectType(IntrospectionType):
    kind: str = "INPUT_OBJECT"
    input_fields: FrozenList["IntrospectionInputValue"]

IntrospectionOutputType = (
    IntrospectionScalarType,
    IntrospectionObjectType,
    IntrospectionInterfaceType,
    IntrospectionUnionType,
    IntrospectionEnumType,
)

IntrospectionInputType = (
    IntrospectionScalarType,
    IntrospectionEnumType,
    IntrospectionInputObjectType,
)

# GraphQL Introspection Type
GIT = TypeVar("GIT", bound="IntrospectionType")

class IntrospectionNamedTypeRef(IntrospectionType, Generic[GIT]):
    kind: GIT["kind"]
    name: str

class IntrospectionListTypeRef(IntrospectionType, Generic[GIT]):
    kind: str = 'LIST'
    of_type: GIT

IntrospectionTypeRef = (
    IntrospectionNamedTypeRef,
    IntrospectionListTypeRef
)

IntrospectionOutputTypeRef = (
    IntrospectionNamedTypeRef[IntrospectionType, IntrospectionOutputType]
)

IntrospectionInputTypeRef = IntrospectionInputType

class IntrospectionField:
    name: str
    description: Optional[str]
    args: FrozenList["IntrospectionInputValue"]
    type: IntrospectionOutputTypeRef
    isDeprecated: bool
    deprecationReason: Optional[str]

class IntrospectionInputValue:
    name: str
    description: Optional[str]
    type: IntrospectionInputTypeRef
    defaultValue: Optional[str]

class IntrospectionEnumValue:
    name: str
    description: Optional[str]
    isDeprecated: bool
    deprecationReason: Optional[str]

#
#   | IntrospectionScalarType
#   | IntrospectionObjectType
#   | IntrospectionInterfaceType
#   | IntrospectionUnionType
#   | IntrospectionEnumType
#   | IntrospectionInputObjectType
#
# IntrospectionOutputType =
#   | IntrospectionScalarType
#   | IntrospectionObjectType
#   | IntrospectionInterfaceType
#   | IntrospectionUnionType
#   | IntrospectionEnumType
#
# IntrospectionInputType =
#   | IntrospectionScalarType
#   | IntrospectionEnumType
#   | IntrospectionInputObjectType


class IntrospectionDirective(TypedDict):
    name: str
    description: Optional[str]
    isRepeatable: Optional[bool]
    locations: FrozenList[DirectiveLocation]
    args: FrozenList[IntrospectionInputValue]


class IntrospectionSchema(TypedDict):
    description: Optional[str]
    queryType: IntrospectionNamedTypeRef[IntrospectionObjectType]
    mutationType: Optional[IntrospectionNamedTypeRef[IntrospectionObjectType]]
    subscriptionType: Optional[IntrospectionNamedTypeRef[IntrospectionObjectType]]
    types: FrozenList[IntrospectionType]
    directives: FrozenList[IntrospectionDirective]


def introspection_from_schema(
    schema: GraphQLSchema,
    descriptions: bool = True,
    specified_by_url: bool = False,
    directive_is_repeatable: bool = True,
    schema_description: bool = True,
) -> IntrospectionSchema:
    """Build an IntrospectionQuery from a GraphQLSchema

    IntrospectionQuery is useful for utilities that care about type and field
    relationships, but do not need to traverse through those relationships.

    This is the inverse of build_client_schema. The primary use case is outside of the
    server context, for instance when doing schema comparisons.
    """
    document = parse(
        get_introspection_query(
            descriptions, specified_by_url, directive_is_repeatable, schema_description
        )
    )

    from ..execution.execute import execute_sync, ExecutionResult

    result = execute_sync(schema, document)
    if not isinstance(result, ExecutionResult):  # pragma: no cover
        raise RuntimeError("Introspection cannot be executed")
    if result.errors:  # pragma: no cover
        raise result.errors[0]
    if not result.data:  # pragma: no cover
        raise GraphQLError("Introspection did not return a result")
    return result.data
