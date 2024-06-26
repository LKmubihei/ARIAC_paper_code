(define (domain ariac_domain)
    (:requirements :strips :typing)
    (:types assemblystation container part robot)
    (:predicates (attach ?part - part ?robot - robot)  (is_enabled ?robot - robot)  (is_reachable ?robot - robot ?container - container)  (on ?part - part ?container - container))
    (:action flip
        :parameters (?robot - robot ?part - part ?container - container)
        :precondition (and (is_enabled ?robot) (is_reachable ?robot ?container) (flip_on ?part ?container))
        :effect (on ?part ?container)
    )
     (:action grasp
        :parameters (?robot - robot ?part - part ?container - container)
        :precondition (and (on ?part ?container) (is_enabled ?robot) (is_reachable ?robot ?container) (not (attach ?part ?robot)))
        :effect (attach ?part ?robot)
    )
     (:action move
        :parameters (?robot - robot ?source_container - container ?destination_container - container)
        :precondition (and (is_enabled ?robot) (is_reachable ?robot ?source_container))
        :effect (and (not (is_reachable ?robot ?source_container)) (is_reachable ?robot ?destination_container))
    )
     (:action place
        :parameters (?robot - robot ?part - part ?container - container)
        :precondition (and (is_enabled ?robot) (is_reachable ?robot ?container) (attach ?part ?robot))
        :effect (and (not (attach ?part ?robot)) (on ?part ?container))
    )
)