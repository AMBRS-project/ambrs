"""ambrs.aero_model -- Base classes for aerosol model integrations"""

class NotImplementedError(Exception):
    """This exception is emitted when a method is called in a class related to
an aerosol model that has not been overridden."""

class Input:
    """ambrs.aero_model.Input -- a base class that defines an interface to be
overridden to process input for an aerosol model."""

    def invocation(self, exe: str, prefix: str) -> str:
        """input.invocation(exe, prefix) -> command string
Override this method to provide the command invoking the aerosol model on the
command line given
    * exe: a path to the aerosol model executable
    * prefix: a prefix that identifies the main input file

You may assume that this command is issued in the directory in which all necessary
input files reside."""
        raise NotImplementedError('Input.invocation not overridden!')

    def write_files(self, dir, prefix) -> None:
        """input.write_files(dir, prefix) -> None
Override this method to write input files for an aerosol model to the given
directory with a main input file identified by the given prefix."""
        raise NotImplementedError('Input.write_files not overridden!')
