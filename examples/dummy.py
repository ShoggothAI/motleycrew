while True:
    did_something = False
    for t in self.tasks:
        matches = t.match_tasks(self.graph)
        if len(matches) > 0:
            t.get_agent().invoke(matches[0])
            did_something = True
    if not did_something:
        break


@dataclass
ToolDescription:
    name: str
    description: str
    framework: str
    toolkit: Optional[str] = None

motley.tool.get_descriptions(framework: Optional[str] = None) -> ToolDescription
motleycrew.tool.from_description(d: ToolDescription) -> MotleyTool
motleycrew.tool.init_embeddings(embed_model) -> ToolSearcher
class ToolSearcher:
    def match(self, task: str) -> List[MotleyTool]:
        pass

    def register_tools(self, ...):
    @classmethod
    def default(cls) -> ToolSearcher:
        pass