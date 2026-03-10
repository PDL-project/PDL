(define (problem move_computer_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    computer - object
    desk - object
    sofa - object
  )
  (:init
    (at robot1 desk)
    (at-location computer desk)
  )
  (:goal
    (and
      (at-location computer sofa)
    )
  )
)