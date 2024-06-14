class SQLExpression(BaseModel):
    # TODO: when creating it, validate that it's a valid pgsql expression!
    # TODO: __str__ should just return expression (for use in prompts)
    expression: str
