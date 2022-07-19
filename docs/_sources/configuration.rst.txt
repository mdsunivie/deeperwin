=====================
Configuration Options
=====================

We store all configuration options in a nested data structure derived from pydantic models.
This provides parsing from configuration files, default management, input validation, and autocompletion
(both for code as well as for config-files).

To get a schema of all possible config options and there defaults refer to `sample_configs/config_schema.json`, which can also be used in IDEs to provide autocompletion when writing config files.

The most common config-options are listed in the table below, follwed by a full documentation of the entire config datastructure:


.. csv-table:: Major configuration options
   :file: major_config_options.csv
   :widths: 15, 25, 60
   :header-rows: 1


.. automodule:: deeperwin.configuration
   :members:
   :exclude-members: ConfigModel, NetworkConfig