from Agent_Nodes import router_node, sql_retriever_node, vector_retriever_node, context_grader_node, query_rewriter_node

print("🧪 Starting Node Tests...\n" + "="*50)

# TEST 1: The Router
print("\n--- TEST 1: Testing Routing Logic ---")
router_node("What was Apple's total revenue in 2024?") # Should pick SQL
router_node("Did the CFO sound worried about memory chip prices during the Q&A?") # Should pick VECTOR
router_node("Compare our actual Q3 revenue against what Tim Cook said about iPhone growth.") # Should pick BOTH

# TEST 2: Vector Retrieval & Grading (The Reflexion Loop in Action!)
print("\n--- TEST 2: Testing Retrieval & Reflexion Grading ---")
question = "What are the exact risks regarding NAND and DRAM chip prices?"

# 1. Retrieve text
context = vector_retriever_node(question)
print(f"\nRetrieved Context Summary:\n{context[:300]}...")

# 2. Grade the text
grade = context_grader_node(question, context)

# 3. Test what happens if we feed it junk data
print("\n--- TEST 3: Testing Grader Rejection & Rewriting ---")
junk_context = "Tim Cook talked about new Apple Watch bands and retail store openings in Mumbai."
bad_grade = context_grader_node(question, junk_context)

if bad_grade == "NO":
    new_query = query_rewriter_node(question)

print("\n" + "="*50 + "\n🎉 All node tests completed! You are ready for Day 2.")