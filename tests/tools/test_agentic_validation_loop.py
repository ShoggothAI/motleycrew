import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from motleycrew.tools.agentic_validation_loop import AgenticValidationLoop


class TestInputSchema(BaseModel):
    test_field: str = Field(description="A test field")


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = "Test response"
    return llm


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.invoke.return_value = {"output": "Test agent response"}
    return agent


class TestAgenticValidationLoop:
    
    def test_init_with_string_prompt(self, mock_llm):
        """Test initialization with a string prompt."""
        tool = AgenticValidationLoop(
            name="test_tool",
            description="A test tool",
            prompt="This is a test prompt with {variable}",
            llm=mock_llm
        )
        
        assert tool.name == "test_tool"
        assert isinstance(tool.prompt_template, PromptTemplate)
        assert "variable" in tool.prompt_template.input_variables
        
        # Check auto-generated schema
        assert hasattr(tool.schema, "model_fields")
        assert "variable" in tool.schema.model_fields
    
    def test_init_with_prompt_template(self, mock_llm):
        """Test initialization with a PromptTemplate."""
        prompt_template = PromptTemplate.from_template("Test {var1} and {var2}")
        
        tool = AgenticValidationLoop(
            name="test_tool",
            description="A test tool",
            prompt=prompt_template,
            llm=mock_llm
        )
        
        assert tool.prompt_template == prompt_template
        assert hasattr(tool.schema, "model_fields")
        assert "var1" in tool.schema.model_fields
        assert "var2" in tool.schema.model_fields
    
    def test_init_with_custom_schema(self, mock_llm):
        """Test initialization with a custom schema."""
        tool = AgenticValidationLoop(
            name="test_tool",
            description="A test tool",
            prompt="Test prompt",
            schema=TestInputSchema,
            llm=mock_llm
        )
        
        assert tool.schema == TestInputSchema
    
    @patch("motleycrew.tools.agentic_validation_loop.ReActToolCallingMotleyAgent")
    @patch("motleycrew.tools.agentic_validation_loop.StructuredPassthroughTool")
    def test_run_method(self, mock_structured_tool, mock_agent_class, mock_llm):
        """Test the run method."""
        # Setup mocks
        mock_tool_instance = MagicMock()
        mock_structured_tool.return_value = mock_tool_instance
        
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke.return_value = {"output": "Test result"}
        mock_agent_class.return_value = mock_agent_instance
        
        # Create tool with template
        tool = AgenticValidationLoop(
            name="test_tool",
            description="A test tool",
            prompt="Test prompt with {variable}",
            llm=mock_llm
        )
        
        # Run the tool
        result = tool.run(variable="test_value")
        
        # Verify prompt was formatted
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["prompt_prefix"] == "Test prompt with test_value"
        
        # Verify agent was invoked
        mock_agent_instance.invoke.assert_called_once_with({})
        assert result == {"output": "Test result"}
    
    def test_post_processing(self, mock_llm):
        """Test post-processing functionality."""
        # Create a post-processing function
        def post_process(result):
            return f"Processed: {result}"
        
        # Setup mocks for agent response
        with patch("motleycrew.tools.agentic_validation_loop.ReActToolCallingMotleyAgent") as mock_agent_class:
            mock_agent_instance = MagicMock()
            mock_agent_instance.invoke.return_value = "Test result"
            mock_agent_class.return_value = mock_agent_instance
            
            # Create tool with post-processing
            tool = AgenticValidationLoop(
                name="test_tool",
                description="A test tool",
                prompt="Test prompt",
                post_process=post_process,
                llm=mock_llm
            )
            
            # Run the tool
            tool.run()
            
            # Verify StructuredPassthroughTool was created with post_process
            with patch("motleycrew.tools.agentic_validation_loop.StructuredPassthroughTool") as mock_structured_tool:
                tool.run()
                call_kwargs = mock_structured_tool.call_args.kwargs
                assert call_kwargs["post_process"] == post_process