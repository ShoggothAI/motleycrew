Customer support chatbot with Ray Serve
=======================================

This example demonstrates how to build a customer support chatbot using MotleyCrew and Ray Serve.
The chatbot is designed to answer customer queries based on a database of past issues and their resolutions.

The code for this example can be found `here <https://github.com/ShoggothAI/motleycrew/tree/main/motleycrew/applications/customer_support>`_.
Also, see the `blog post <https://blog.motleycrew.ai/blog/building-a-customer-support-chatbot-using-motleycrew-and-ray>`_ about this app.

Key Components
--------------

1. Issue Database

   - Stores information about past issues and their solutions in a tree structure
   - Intermediate nodes represent issue categories
   - Leaf nodes represent individual issues
   - Uses Kuzu to store and query the issue tree through our OGM (see :doc:`../knowledge_graph` for more details)

2. AI Support Agent

   - Attempts to resolve customer issues based on past solutions
   - Navigates the issue tree to find relevant information
   - Can ask clarifying questions to the customer
   - Proposes solutions or escalates to a human agent if necessary

3. Agent Tools

   - IssueTreeViewTool: Allows the agent to navigate the issue tree
   - CustomerChatTool: Enables the agent to ask additional questions to the customer
   - ResolveIssueTool: Used to submit a solution or escalate to a human agent

4. Ray Serve Deployment

   - Exposes the chatbot as an API
   - Allows multiple customers to connect simultaneously
   - Uses WebSockets over FastAPI for communication

Implementation Details
----------------------

The support agent is implemented using the "every response is a tool call" design.
The agent loop can only end with a ResolveIssueTool call or when a constraint (e.g., number of iterations) is reached.
This is achieved by making the ResolveIssueTool an output handler.

The Ray Serve deployment is configured using a simple decorator:

.. code-block:: python

    @serve.deployment(num_replicas=3, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
    class SupportAgentDeployment:
        ...

This setup allows for easy scaling and supports multiple simultaneous sessions balanced between replicas.

Running the Example
-------------------

The project includes sample issue data that can be used to populate the issue tree.

To run this example:

.. code-block:: bash

    git clone https://github.com/ShoggothAI/motleycrew.git
    cd motleycrew
    pip install -r requirements.txt

    python -m motleycrew.applications.customer_support.issue_tree  # populate the issue tree
    ray start --head
    python -m motleycrew.applications.customer_support.ray_serve_app

This example showcases the flexibility of MotleyCrew for building agent-based applications, allowing you to choose your preferred agent framework, orchestration model, and deployment solution.
