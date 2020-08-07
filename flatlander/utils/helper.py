
class Dynamic(object):
    """
    Does not throw when accessing non-existing attribute
    """
    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except Exception as _:
            return None
