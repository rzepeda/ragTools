#!/usr/bin/env python3
"""Direct test of agentic tools to verify database access."""

import asyncio
import sys
sys.path.insert(0, '/mnt/MCPProyects/ragTools')

from dotenv import load_dotenv
load_dotenv('/mnt/MCPProyects/ragTools/.env')

from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.config.strategy_pair_manager import StrategyPairManager

async def test():
    print("=" * 60)
    print("Testing Agentic Tools Database Access")
    print("=" * 60)
    
    # Load services
    registry = ServiceRegistry.from_config('config/services.yaml')
    manager = StrategyPairManager(registry)
    
    print("\n1. Loading agentic strategy...")
    indexer, retriever = manager.load_pair('agentic-rag-pair')
    print("✅ Strategy loaded")
    
    # Get the semantic search tool
    print("\n2. Getting semantic_search tool...")
    tools = retriever.tools
    semantic_tool = next((t for t in tools if t.name == "semantic_search"), None)
    
    if not semantic_tool:
        print("❌ semantic_search tool not found!")
        return
    
    print(f"✅ Found tool: {semantic_tool.name}")
    
    # Try to execute the tool directly
    print("\n3. Executing semantic_search tool...")
    try:
        result = await semantic_tool.execute(query="Voyager 1", top_k=3)
        print(f"✅ Tool executed successfully!")
        print(f"   Success: {result.success}")
        print(f"   Data count: {len(result.data) if isinstance(result.data, list) else 0}")
        print(f"   Error: {result.error}")
        
        if result.success and result.data:
            print(f"\n4. Sample result:")
            first = result.data[0]
            if isinstance(first, dict):
                print(f"   Text: {first.get('text', 'N/A')[:100]}...")
                print(f"   Score: {first.get('score', 'N/A')}")
    except Exception as e:
        print(f"❌ Tool execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test())
