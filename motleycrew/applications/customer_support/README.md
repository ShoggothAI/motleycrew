# Customer support agent demo

This is a demo of a customer support app built using motleycrew and Ray.

It includes sample data for populating the issue tree.


## Installation and usage
We suggest you set up a virtualenv for managing the environment.

```
git clone https://github.com/ShoggothAI/motleycrew.git
cd motleycrew
pip install -r requirements.txt

python -m motleycrew.applications.customer_support.issue_tree  # populate the issue tree
ray start --head
python -m motleycrew.applications.customer_support.ray_serve_app
```

Navigate to http://127.0.0.1:8000/ and have fun!
Also, check out the Ray dashboard for the app logs etc.

## Example screenshot
<img width="923" alt="image" src="https://github.com/user-attachments/assets/f55be49d-0eb4-4cdd-ba2b-ee75f5468371">
