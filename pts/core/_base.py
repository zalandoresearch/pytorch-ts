def fqname_for(cls: type) -> str:
    """
    Returns the fully qualified name of ``cls``.

    Parameters
    ----------
    cls
        The class we are interested in.

    Returns
    -------
    str
        The fully qualified name of ``cls``.
    """
    return f"{cls.__module__}.{cls.__qualname__}"