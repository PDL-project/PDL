(define (problem move_spoon_to_drawer_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    spoon - object
    drawer - object
    counterTop - object
  )
  (:init
    (at robot2 counterTop)
    (at-location spoon counterTop)
    (at-location drawer counterTop) ; This line is not necessary but doesn't affect validity.
  )
  (:goal
    (and
      (at-location spoon drawer)
    )
  )
)