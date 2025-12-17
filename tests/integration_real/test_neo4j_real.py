"""
Real integration tests for Neo4j graph database service.

Tests actual graph database operations with real Neo4j instance configured via .env.
"""

import pytest


@pytest.mark.real_integration
@pytest.mark.requires_neo4j
@pytest.mark.asyncio
async def test_neo4j_connection(real_neo4j_service):
    """Test that we can connect to Neo4j."""
    assert real_neo4j_service is not None
    
    # Test basic query
    result = await real_neo4j_service.execute_query("RETURN 1 as test")
    assert result is not None


@pytest.mark.real_integration
@pytest.mark.requires_neo4j
@pytest.mark.asyncio
async def test_create_and_retrieve_node(real_neo4j_service):
    """Test creating and retrieving a node."""
    # Create a test node
    await real_neo4j_service.execute_query(
        "CREATE (n:TestNode {name: $name, value: $value})",
        {"name": "test1", "value": 42}
    )
    
    # Retrieve the node
    result = await real_neo4j_service.execute_query(
        "MATCH (n:TestNode {name: $name}) RETURN n.name as name, n.value as value",
        {"name": "test1"}
    )
    
    assert result is not None
    assert len(result) > 0
    assert result[0]["name"] == "test1"
    assert result[0]["value"] == 42


@pytest.mark.real_integration
@pytest.mark.requires_neo4j
@pytest.mark.asyncio
async def test_create_relationship(real_neo4j_service):
    """Test creating relationships between nodes."""
    # Create two nodes with a relationship
    await real_neo4j_service.execute_query("""
        CREATE (a:TestNode {name: 'Alice'})
        CREATE (b:TestNode {name: 'Bob'})
        CREATE (a)-[:KNOWS {since: 2020}]->(b)
    """)
    
    # Query the relationship
    result = await real_neo4j_service.execute_query("""
        MATCH (a:TestNode {name: 'Alice'})-[r:KNOWS]->(b:TestNode)
        RETURN a.name as from, b.name as to, r.since as since
    """)
    
    assert len(result) > 0
    assert result[0]["from"] == "Alice"
    assert result[0]["to"] == "Bob"
    assert result[0]["since"] == 2020


@pytest.mark.real_integration
@pytest.mark.requires_neo4j
@pytest.mark.asyncio
async def test_graph_traversal(real_neo4j_service):
    """Test graph traversal queries."""
    # Create a small graph: A -> B -> C
    await real_neo4j_service.execute_query("""
        CREATE (a:TestNode {name: 'A'})
        CREATE (b:TestNode {name: 'B'})
        CREATE (c:TestNode {name: 'C'})
        CREATE (a)-[:CONNECTS_TO]->(b)
        CREATE (b)-[:CONNECTS_TO]->(c)
    """)
    
    # Find path from A to C
    result = await real_neo4j_service.execute_query("""
        MATCH path = (a:TestNode {name: 'A'})-[:CONNECTS_TO*]->(c:TestNode {name: 'C'})
        RETURN length(path) as pathLength
    """)
    
    assert len(result) > 0
    assert result[0]["pathLength"] == 2  # A -> B -> C


@pytest.mark.real_integration
@pytest.mark.requires_neo4j
@pytest.mark.asyncio
async def test_entity_storage(real_neo4j_service):
    """Test storing entities with properties."""
    # Store an entity with multiple properties
    entity_data = {
        "id": "entity_1",
        "type": "Person",
        "name": "John Doe",
        "age": 30,
        "occupation": "Engineer"
    }
    
    await real_neo4j_service.execute_query(
        "CREATE (e:Entity:Person {id: $id, name: $name, age: $age, occupation: $occupation})",
        entity_data
    )
    
    # Retrieve the entity
    result = await real_neo4j_service.execute_query(
        "MATCH (e:Entity {id: $id}) RETURN e",
        {"id": "entity_1"}
    )
    
    assert len(result) > 0
    entity = result[0]["e"]
    assert entity["name"] == "John Doe"
    assert entity["age"] == 30
    assert entity["occupation"] == "Engineer"


@pytest.mark.real_integration
@pytest.mark.requires_neo4j
@pytest.mark.asyncio
async def test_batch_node_creation(real_neo4j_service):
    """Test creating multiple nodes in batch."""
    # Create multiple nodes
    nodes = [
        {"name": f"Node{i}", "value": i}
        for i in range(10)
    ]
    
    for node in nodes:
        await real_neo4j_service.execute_query(
            "CREATE (n:TestNode {name: $name, value: $value})",
            node
        )
    
    # Count created nodes
    result = await real_neo4j_service.execute_query(
        "MATCH (n:TestNode) WHERE n.name STARTS WITH 'Node' RETURN count(n) as count"
    )
    
    assert result[0]["count"] >= 10


@pytest.mark.real_integration
@pytest.mark.requires_neo4j
@pytest.mark.asyncio
async def test_update_node_properties(real_neo4j_service):
    """Test updating node properties."""
    # Create a node
    await real_neo4j_service.execute_query(
        "CREATE (n:TestNode {name: 'UpdateTest', value: 1})"
    )
    
    # Update the node
    await real_neo4j_service.execute_query(
        "MATCH (n:TestNode {name: 'UpdateTest'}) SET n.value = 100, n.updated = true"
    )
    
    # Verify update
    result = await real_neo4j_service.execute_query(
        "MATCH (n:TestNode {name: 'UpdateTest'}) RETURN n.value as value, n.updated as updated"
    )
    
    assert result[0]["value"] == 100
    assert result[0]["updated"] is True


@pytest.mark.real_integration
@pytest.mark.requires_neo4j
@pytest.mark.asyncio
async def test_delete_nodes(real_neo4j_service):
    """Test deleting nodes."""
    # Create a node
    await real_neo4j_service.execute_query(
        "CREATE (n:TestNode {name: 'DeleteTest'})"
    )
    
    # Verify it exists
    result = await real_neo4j_service.execute_query(
        "MATCH (n:TestNode {name: 'DeleteTest'}) RETURN count(n) as count"
    )
    assert result[0]["count"] == 1
    
    # Delete the node
    await real_neo4j_service.execute_query(
        "MATCH (n:TestNode {name: 'DeleteTest'}) DELETE n"
    )
    
    # Verify it's deleted
    result = await real_neo4j_service.execute_query(
        "MATCH (n:TestNode {name: 'DeleteTest'}) RETURN count(n) as count"
    )
    assert result[0]["count"] == 0


@pytest.mark.real_integration
@pytest.mark.requires_neo4j
@pytest.mark.asyncio
async def test_complex_query(real_neo4j_service):
    """Test complex query with aggregation and filtering."""
    # Create test data
    await real_neo4j_service.execute_query("""
        CREATE (p1:TestNode:Product {name: 'Product1', price: 100, category: 'Electronics'})
        CREATE (p2:TestNode:Product {name: 'Product2', price: 200, category: 'Electronics'})
        CREATE (p3:TestNode:Product {name: 'Product3', price: 50, category: 'Books'})
    """)
    
    # Query with aggregation
    result = await real_neo4j_service.execute_query("""
        MATCH (p:Product)
        WHERE p.category = 'Electronics'
        RETURN avg(p.price) as avgPrice, count(p) as count
    """)
    
    assert result[0]["avgPrice"] == 150.0
    assert result[0]["count"] == 2
