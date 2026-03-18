from agents import Agent, WebSearchTool, Runner
from agents.model_settings import ModelSettings
import json

instructions = """
You are a friendly book recommendation assistant for a bookstore.

You help customers find the perfect book based on:
- Child's age
- Reading level (beginner, intermediate, advanced)
- Interests
- Budget (in Indian Rupees)

You will be given a list of available books from the store inventory.
You MUST only recommend books from this list.
Never suggest books not in the inventory.

If no books match perfectly, recommend the closest available option and explain why.

Always be warm, friendly and helpful.
Ask for missing information naturally if not provided.

When you have enough info, recommend books in this format:
- Book title
- Why it's perfect for this child
- Price

Keep responses concise and conversational.
"""

recommendation_agent = Agent(
    name="book_recommendation_agent",
    instructions=instructions,
    model="gpt-4o"
)

async def get_recommendations(conversation_history, inventory):
    # Build system context with inventory
    inventory_text = "Available books in store:\n"
    for book in inventory:
        price = book.get('price', 'Price not listed')
        name = book.get('name', '')
        description = book.get('description', '')
        inventory_text += f"- {name} | Price: ₹{price} | {description[:100]}\n"

    # Inject inventory into first message context
    messages = [
        {
            "role": "user",
            "content": f"[STORE INVENTORY]\n{inventory_text}\n\n[CUSTOMER MESSAGE]\n{conversation_history[-1]['content']}"
        }
    ]

    # Add conversation history except last message
    if len(conversation_history) > 1:
        messages = conversation_history[:-1] + messages

    result = await Runner.run(recommendation_agent, messages)
    return result.final_output