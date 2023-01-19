

class Registry(object):
    """Registry 

    References:
        [1] https://github.com/google-research/pix2seq/blob/46a7531d356bd6353e748fad07299b15fd92cfb4/registry.py
    """

    def __init__(self):
        self.registry = {}


    def register(self, key):
        """Returns callable to register value for key.
        Args: 
            key (str)
        Returns:
            _register (callable function)
        """

        def _register(item):
            if key in self.registry:
                raise ValueError(f"{key} is already registered!")

            self.registry[key] = item
            return item
        return _register 

    def lookup(self, key):
        """Look up value for key.
        Args:
            key (str)
        
        Returns:
            item 
        """

        if key not in self.registry:
            valid_keys = "\n".join(self.registry.keys())

            raise ValueError(
                f"{key} is not registered! \n\n"
                f"Valid keys: {valid_keys} \n\n"
            )

        return self.registry[key]    
