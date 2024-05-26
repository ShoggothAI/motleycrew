Installation
============

Installation using pip
---------------------

.. code-block:: console

    pip install motleycrew

Installation from source
------------------------
| Motleycrew uses Poetry to manage its dependencies. We suggest you use it if you want to install motleycrew from source.
| For installation instructions, see https://python-poetry.org/docs/#installation.

Clone the repository_ and install the dependencies:

.. code-block:: console

    git clone https://github.com/ShoggothAI/motleycrew.git
    cd motleycrew
    poetry install

This will create a virtual environment and install the required dependencies.

You might also want to install optional dependencies (extra libraries like LlamaIndex or CrewAI) for additional functionality:

.. code-block:: console

    poetry install --all-extras

If you want to install extra dependencies for development, you can use the following command:

.. code-block:: console

    poetry install --all-extras --with dev

.. _repository: https://github.com/ShoggothAI/motleycrew
