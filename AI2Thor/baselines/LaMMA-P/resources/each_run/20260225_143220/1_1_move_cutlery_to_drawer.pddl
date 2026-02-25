(define (problem move_cutlery_to_drawer)
  (:domain robot1)
  (:objects
    robot1 - robot
    fork - object
    spoon - object
    knife - object
    diningTable - object
    drawer - object
  )
  (:init
    (at robot1 diningTable)
    (at-location fork diningTable)
    (at-location spoon diningTable)
    (at-location knife diningTable)
  )
  (:goal
    (and
      (at-location fork drawer)
      (at-location spoon drawer)
      (at-location knife drawer)
    )
  )
)