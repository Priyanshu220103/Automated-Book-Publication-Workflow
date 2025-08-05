from graphviz import Digraph

# Create a Digraph object
dot = Digraph(comment="LangGraph Workflow")

# Define nodes
dot.node("Scrape", "Scrape")
dot.node("Write", "Write")
dot.node("Review", "Review")
dot.node("Reward", "Reward")
dot.node("Version", "Version")
dot.node("HumanLoop", "HumanLoop")
dot.node("Voice", "Voice")
dot.node("END", "END", shape="doublecircle")

# Entry point
dot.node("ENTRY", "ENTRY POINT", shape="circle")
dot.edge("ENTRY", "Scrape")

# Define edges
dot.edge("Scrape", "Write")
dot.edge("Write", "Review")
dot.edge("Review", "Reward")
dot.edge("Reward", "Version")
dot.edge("Version", "HumanLoop")

# Conditional edges from HumanLoop
dot.edge("HumanLoop", "Voice", label='intent = "stop"')
dot.edge("HumanLoop", "Write", label='intent = "improve"')

# Final edge to END
dot.edge("Voice", "END")

# Render or view
dot.render("Testing_for_my_self\langgraph_workflow", format="png", cleanup=True)  
