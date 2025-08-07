# Coordination Memory Module

The Coordination Memory module manages multi-agent communication and collaboration within the MemoRizz library. This specialized memory system enables multiple AI agents to share information, coordinate tasks, and collaborate effectively through shared blackboard architectures.

## Overview

Coordination memory is distinct from traditional cognitive memory types as it serves inter-agent rather than intra-agent functions. It enables:
- **Multi-agent communication**: Shared information exchange between multiple agents
- **Task coordination**: Synchronization and orchestration of collaborative work
- **Shared context**: Common understanding and shared state across agent networks
- **Blackboard architecture**: Centralized coordination hub for distributed intelligence

## When to Use Coordination Memory

Use coordination memory when you need to:
- Enable multiple agents to work together on complex tasks
- Share information and context between different AI agents
- Coordinate task delegation and result integration
- Implement multi-agent workflows and orchestration
- Create hierarchical agent architectures (root + delegate agents)
- Maintain shared state across distributed agent systems

**Examples of coordination memory content:**
- "Agent A completed data analysis, results available for Agent B"
- "Task delegation: Agent C assigned to handle customer inquiries"
- "Shared context: Current project status and next steps"
- "Inter-agent communication log for troubleshooting"

## Core Features

- **Blackboard Communication**: Centralized communication hub for agent coordination
- **Session Management**: Create and manage multi-agent collaboration sessions
- **Task Delegation**: Support for hierarchical agent structures and task assignment
- **Shared Context**: Maintain common understanding across agent networks
- **Communication Logging**: Track inter-agent interactions for transparency
- **Real-time Coordination**: Enable dynamic task assignment and result sharing

## Usage

### Basic Usage (Without MemAgent)

```python
from memorizz.coordination import SharedMemory
from memorizz.memory_provider.mongodb import MongoDBProvider

# Initialize a memory provider
memory_provider = MongoDBProvider({
    "connection_string": "mongodb://localhost:27017",
    "database_name": "memorizz_db"
})

# === SHARED MEMORY SETUP ===
# Create shared memory for multi-agent coordination
shared_memory = SharedMemory(memory_provider)

# Create a coordination session
root_agent_id = "orchestrator_agent"
delegate_agent_ids = ["research_agent", "analysis_agent", "writing_agent"]

session_id = shared_memory.create_shared_session(
    root_agent_id=root_agent_id,
    delegate_agent_ids=delegate_agent_ids
)

print(f"Created coordination session: {session_id}")

# === BASIC COORDINATION COMMUNICATION ===
# Root agent posts a task assignment
shared_memory.post_message(
    session_id=session_id,
    agent_id=root_agent_id,
    content={
        "task": "research_assignment",
        "target_agent": "research_agent", 
        "details": "Research market trends for Q4 2024",
        "deadline": "2024-12-15"
    },
    entry_type="task_assignment"
)

# Delegate agent acknowledges task
shared_memory.post_message(
    session_id=session_id,
    agent_id="research_agent",
    content={
        "status": "acknowledged",
        "task_id": "research_assignment",
        "estimated_completion": "2024-12-10"
    },
    entry_type="task_acknowledgment"
)

# Delegate agent posts progress update
shared_memory.post_message(
    session_id=session_id,
    agent_id="research_agent", 
    content={
        "task_id": "research_assignment",
        "progress": "50%",
        "findings": "Initial data collected, analyzing trends",
        "status": "in_progress"
    },
    entry_type="progress_update"
)

# === RETRIEVING COORDINATION HISTORY ===
# Get all messages for the session
session_history = shared_memory.get_session_history(session_id)
print(f"Session has {len(session_history)} coordination messages")

# Get messages from specific agent
research_messages = shared_memory.get_agent_messages(session_id, "research_agent")
print(f"Research agent has posted {len(research_messages)} messages")

# Get messages by type
task_assignments = shared_memory.get_messages_by_type(session_id, "task_assignment")
print(f"Found {len(task_assignments)} task assignments")
```

### Advanced Multi-Agent Coordination

```python
import time
from typing import Dict, List, Any

class MultiAgentOrchestrator:
    """Advanced orchestration system for multi-agent coordination."""
    
    def __init__(self, memory_provider):
        self.shared_memory = SharedMemory(memory_provider)
        self.active_sessions = {}
        self.agent_capabilities = {}
    
    def setup_project_coordination(self, project_name: str, agents: Dict[str, List[str]]):
        """Set up coordination for a multi-agent project."""
        
        # Create coordination session
        session_id = self.shared_memory.create_shared_session(
            root_agent_id="project_manager",
            delegate_agent_ids=list(agents.keys())
        )
        
        # Register agent capabilities
        for agent_id, capabilities in agents.items():
            self.agent_capabilities[agent_id] = capabilities
            
            # Post agent capabilities to shared memory
            self.shared_memory.post_message(
                session_id=session_id,
                agent_id=agent_id,
                content={
                    "agent_id": agent_id,
                    "capabilities": capabilities,
                    "status": "available"
                },
                entry_type="agent_registration"
            )
        
        self.active_sessions[project_name] = session_id
        return session_id
    
    def delegate_task(self, project_name: str, task: Dict[str, Any]):
        """Intelligently delegate tasks based on agent capabilities."""
        session_id = self.active_sessions[project_name]
        
        # Find best agent for the task
        best_agent = self._find_best_agent(task["required_capabilities"])
        
        if best_agent:
            # Assign task
            self.shared_memory.post_message(
                session_id=session_id,
                agent_id="project_manager",
                content={
                    "task_id": task["task_id"],
                    "assigned_to": best_agent,
                    "description": task["description"],
                    "requirements": task["required_capabilities"],
                    "priority": task.get("priority", "medium"),
                    "deadline": task.get("deadline")
                },
                entry_type="task_delegation"
            )
            
            return best_agent
        else:
            # No suitable agent found
            self.shared_memory.post_message(
                session_id=session_id,
                agent_id="project_manager",
                content={
                    "task_id": task["task_id"],
                    "status": "unassigned",
                    "reason": "no_suitable_agent",
                    "required_capabilities": task["required_capabilities"]
                },
                entry_type="task_assignment_failed"
            )
            return None
    
    def _find_best_agent(self, required_capabilities: List[str]) -> str:
        """Find the best agent for given capabilities."""
        best_agent = None
        best_match_score = 0
        
        for agent_id, agent_caps in self.agent_capabilities.items():
            # Calculate capability match score
            matches = len(set(required_capabilities) & set(agent_caps))
            match_score = matches / len(required_capabilities)
            
            if match_score > best_match_score:
                best_match_score = match_score
                best_agent = agent_id
        
        return best_agent if best_match_score > 0.5 else None
    
    def collect_results(self, project_name: str, task_id: str) -> Dict:
        """Collect and integrate results from multiple agents."""
        session_id = self.active_sessions[project_name]
        
        # Get all result messages for the task
        all_messages = self.shared_memory.get_session_history(session_id)
        task_results = [
            msg for msg in all_messages 
            if msg.get('content', {}).get('task_id') == task_id 
            and msg.get('entry_type') == 'task_result'
        ]
        
        # Integrate results
        integrated_result = {
            "task_id": task_id,
            "contributions": {},
            "final_status": "completed",
            "integration_timestamp": time.time()
        }
        
        for result in task_results:
            agent_id = result['agent_id']
            integrated_result["contributions"][agent_id] = result['content']
        
        # Post integrated result
        self.shared_memory.post_message(
            session_id=session_id,
            agent_id="project_manager",
            content=integrated_result,
            entry_type="integrated_result"
        )
        
        return integrated_result

# Use advanced orchestration
orchestrator = MultiAgentOrchestrator(memory_provider)

# Set up project with specialized agents
project_agents = {
    "data_analyst": ["data_analysis", "statistics", "visualization"],
    "content_writer": ["writing", "editing", "communication"],
    "researcher": ["research", "information_gathering", "synthesis"],
    "technical_expert": ["programming", "system_design", "debugging"]
}

session_id = orchestrator.setup_project_coordination("market_research_project", project_agents)

# Delegate tasks intelligently
research_task = {
    "task_id": "market_analysis_001",
    "description": "Analyze Q4 market trends in technology sector",
    "required_capabilities": ["research", "data_analysis"],
    "priority": "high",
    "deadline": "2024-12-20"
}

assigned_agent = orchestrator.delegate_task("market_research_project", research_task)
print(f"Task assigned to: {assigned_agent}")
```

### Using with MemAgent

```python
from memorizz.memagent import MemAgent
from memorizz.multi_agent_orchestrator import MultiAgentOrchestrator

# === SETTING UP COORDINATED AGENTS ===
# Create root orchestrator agent
root_agent = MemAgent(
    memory_provider=memory_provider,
    application_mode="workflow",
    instruction="You are a project orchestrator managing multiple specialized agents."
)

# Create specialized delegate agents
research_agent = MemAgent(
    memory_provider=memory_provider,
    application_mode="deep_research",
    instruction="You are a research specialist focused on data gathering and analysis."
)

analysis_agent = MemAgent(
    memory_provider=memory_provider,
    application_mode="workflow", 
    instruction="You are an analysis expert specializing in data interpretation."
)

writing_agent = MemAgent(
    memory_provider=memory_provider,
    application_mode="assistant",
    instruction="You are a content creation specialist for reports and documentation."
)

# === COORDINATION SETUP ===
# Create shared memory session for coordination
shared_memory = SharedMemory(memory_provider)
session_id = shared_memory.create_shared_session(
    root_agent_id=root_agent.agent_id,
    delegate_agent_ids=[
        research_agent.agent_id,
        analysis_agent.agent_id, 
        writing_agent.agent_id
    ]
)

# Attach shared memory to agents
for agent in [root_agent, research_agent, analysis_agent, writing_agent]:
    agent.shared_memory_session_id = session_id

# === COORDINATED TASK EXECUTION ===
# Root agent initiates project
project_brief = "Create a comprehensive market research report on AI trends"

# Root agent delegates research
root_response = root_agent.run(f"Delegate research task: {project_brief}")
# Root agent automatically posts delegation to shared memory

# Research agent receives and processes task
research_result = research_agent.run("Check for any assigned research tasks and complete them")
# Research agent reads from shared memory, completes task, posts results

# Analysis agent processes research results
analysis_result = analysis_agent.run("Analyze any available research data from the shared workspace")
# Analysis agent reads research results from shared memory, performs analysis

# Writing agent creates final report
final_report = writing_agent.run("Create a report using all available research and analysis")
# Writing agent integrates all shared memory content into final deliverable

print("Coordinated multi-agent project completed")
print(f"Final report: {final_report[:200]}...")

# === MONITORING COORDINATION ===
# Monitor coordination session
session_history = shared_memory.get_session_history(session_id)
print(f"\\nCoordination session had {len(session_history)} interactions")

for message in session_history[-5:]:  # Last 5 messages
    print(f"{message['agent_id']}: {message['entry_type']} - {message['created_at']}")
```

### Advanced Coordination Patterns

```python
class HierarchicalCoordination:
    """Implement hierarchical agent coordination patterns."""
    
    def __init__(self, memory_provider):
        self.shared_memory = SharedMemory(memory_provider)
        self.hierarchy = {}
    
    def setup_hierarchy(self, structure: Dict):
        """Set up hierarchical agent structure."""
        # Example structure:
        # {
        #     "CEO": ["department_head_1", "department_head_2"],
        #     "department_head_1": ["specialist_1", "specialist_2"],
        #     "department_head_2": ["specialist_3", "specialist_4"]
        # }
        
        self.hierarchy = structure
        
        # Create coordination sessions for each level
        sessions = {}
        
        for manager, subordinates in structure.items():
            session_id = self.shared_memory.create_shared_session(
                root_agent_id=manager,
                delegate_agent_ids=subordinates
            )
            sessions[manager] = session_id
        
        return sessions
    
    def cascade_communication(self, message: str, start_level: str = "CEO"):
        """Cascade communication down the hierarchy."""
        if start_level not in self.hierarchy:
            return
        
        # Send to immediate subordinates
        subordinates = self.hierarchy[start_level]
        session_id = self.sessions.get(start_level)
        
        if session_id:
            self.shared_memory.post_message(
                session_id=session_id,
                agent_id=start_level,
                content={
                    "message": message,
                    "type": "cascade_communication",
                    "level": start_level
                },
                entry_type="hierarchical_communication"
            )
        
        # Recursively cascade to lower levels
        for subordinate in subordinates:
            if subordinate in self.hierarchy:
                self.cascade_communication(message, subordinate)

# Specialized coordination for different scenarios
class ResearchCoordination:
    """Specialized coordination for research workflows."""
    
    def __init__(self, memory_provider):
        self.shared_memory = SharedMemory(memory_provider)
    
    def coordinate_research_pipeline(self, research_query: str):
        """Coordinate a complete research pipeline."""
        
        # Set up research team
        session_id = self.shared_memory.create_shared_session(
            root_agent_id="research_coordinator",
            delegate_agent_ids=["data_collector", "analyst", "synthesizer", "reviewer"]
        )
        
        # Phase 1: Data Collection
        self.shared_memory.post_message(
            session_id=session_id,
            agent_id="research_coordinator",
            content={
                "phase": "data_collection",
                "query": research_query,
                "assigned_to": "data_collector",
                "requirements": ["comprehensive_search", "source_verification"]
            },
            entry_type="phase_assignment"
        )
        
        # Coordination continues through each phase...
        return session_id
```

## Memory Integration

### Multi-Agent Architecture

Coordination memory works alongside individual agent memory systems:

```python
# Each agent maintains its own memory systems
agent_a = MemAgent(memory_provider=memory_provider)  # Has semantic, episodic, procedural memory
agent_b = MemAgent(memory_provider=memory_provider)  # Has semantic, episodic, procedural memory

# Coordination memory enables communication between them
shared_session = shared_memory.create_shared_session(
    root_agent_id=agent_a.agent_id,
    delegate_agent_ids=[agent_b.agent_id]
)

# Agents can now coordinate while maintaining individual memories
```

### Coordination vs Cognitive Memory

```python
# Cognitive memory (individual agent):
# - Semantic: What the agent knows
# - Episodic: What the agent has experienced  
# - Procedural: What the agent can do
# - Working: What the agent is currently processing

# Coordination memory (multi-agent):
# - Shared: What agents need to communicate
# - Delegated: Task assignments and responsibilities
# - Collective: Shared understanding and context
# - Collaborative: Joint problem-solving efforts
```

## Implementation Notes

### Blackboard Architecture
- Centralized communication hub for all agents in a session
- Message-based communication with typed entries
- Persistent storage of coordination history
- Real-time coordination capabilities

### Session Management
- Multiple concurrent coordination sessions
- Session-based access control and privacy
- Hierarchical agent organization support
- Session lifecycle management

### Communication Patterns
- Task delegation and result collection
- Status updates and progress tracking
- Context sharing and state synchronization
- Error handling and conflict resolution

### Best Practices
- Clear agent role definition and capability mapping
- Structured message formats for different coordination types
- Regular session monitoring and cleanup
- Coordination pattern documentation
- Multi-agent testing and validation

### Performance Considerations
- Efficient message routing and filtering
- Session-based memory isolation
- Background coordination processing
- Communication load balancing

This coordination memory system enables sophisticated multi-agent architectures where individual AI agents can collaborate effectively while maintaining their own cognitive memory systems and specialized capabilities.