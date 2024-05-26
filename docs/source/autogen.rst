AutoGen-related Examples
========================

Here are some examples that firstly, show how some AutoGen patterns translate into motleycrew (in particular,
how cases where UserProxy is only used as an AgentExecutor don't need multiple agents in other frameworks),
and secondly, how to use motleycrew together with autogen, both by wrapping a collection of autogen agents as
a motleycrew tool, and by giving motleycrew tools and agents as tools to autogen.

.. toctree::
   :maxdepth: 2

   examples/math_single_agent
   examples/integrating_autogen
