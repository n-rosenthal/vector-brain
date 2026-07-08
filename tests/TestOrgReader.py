from src.extractors import connect_readonly, fetch_nodes, nodes_with_text


def test_fetch_nodes_reads_tags_and_titles(fake_org_roam_db):
    conn = connect_readonly(fake_org_roam_db)
    nodes = fetch_nodes(conn)

    assert len(nodes) == 3
    by_id = {n.node_id: n for n in nodes}
    assert by_id["node-python"].title == "Python"
    assert by_id["node-python"].tags == ["programacao"]
    assert by_id["node-rust"].tags == []  # sem tag associada


def test_extract_subtree_excludes_headline_and_properties(fake_org_roam_db):
    conn = connect_readonly(fake_org_roam_db)
    results = {node.node_id: text for node, text in nodes_with_text(conn)}

    python_text = results["node-python"]
    assert "* Python" not in python_text
    assert "PROPERTIES" not in python_text
    assert ":ID:" not in python_text
    assert "Python é uma linguagem" in python_text


def test_extract_subtree_stops_at_next_headline(fake_org_roam_db):
    """O texto do node Python não pode vazar pro conteúdo do node Rust."""
    conn = connect_readonly(fake_org_roam_db)
    results = {node.node_id: text for node, text in nodes_with_text(conn)}

    assert "Rust" not in results["node-python"]
    assert "garbage collector" not in results["node-python"]


def test_extract_subtree_last_node_goes_to_end_of_file(fake_org_roam_db):
    conn = connect_readonly(fake_org_roam_db)
    results = {node.node_id: text for node, text in nodes_with_text(conn)}

    assert "sem garbage collector" in results["node-rust"]
