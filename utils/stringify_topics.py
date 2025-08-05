def stringify_topics(topics : list[str]) -> str:
    return ", ".join(topics[:-1])+" and "+topics[-1]