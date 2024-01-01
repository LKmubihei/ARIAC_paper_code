#!/usr/bin/env python3

import py_trees as pt

class LoggingVisitor(pt.visitors.VisitorBase):
    def __init__(self):
        self.log = []

    def visit(self, behaviour):
        self.log.append(f"Visiting node: {behaviour.name}")

if __name__ == "__main__":
    # 创建行为树
    root = pt.composites.Sequence(name="Root",memory=True)
    selector = pt.composites.Selector(name="Selector",memory=True)
    action = pt.behaviours.Success(name="Action")

    root.add_child(selector)
    selector.add_child(action)

    # 创建 Visitors 并将其添加到行为树
    visitor = LoggingVisitor()
    behaviour_tree = pt.trees.BehaviourTree(root)
    behaviour_tree.visitors.append(visitor)

    # 执行行为树
    try:
        behaviour_tree.tick_tock(period_ms=1000)
    except KeyboardInterrupt:
        behaviour_tree.interrupt()

    # 打印日志
    for log_entry in visitor.log:
        print(log_entry)
