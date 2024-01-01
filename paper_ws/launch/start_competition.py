#!/usr/bin/env python3

import rclpy
import time
import threading
from rclpy.executors import MultiThreadedExecutor,SingleThreadedExecutor
from paper_ws.paper import PaperInterface




def main(args=None):
    rclpy.init(args=args)


    interface = PaperInterface()
    
    interface.new_thread()
    
    interface.wait(2)
    
    interface.start_competition()
    
    # interface.wait(2)

    interface.run()
    


    interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

