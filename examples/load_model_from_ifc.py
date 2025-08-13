from hierarchical.abstractions import Model

model = Model.from_ifc(ifc_file="/Users/benjaminfriedman/repos/hierArchical/sample_house.ifc")

# Example usage
# view the spaces + objects
model.show_spaces()
model.show_objects()

# view the full building graph
model.show_building_graph()

# Example Ask
print("Q: What spaces are in the model?")
print("A:", model.ask("What spaces are in the model?"))