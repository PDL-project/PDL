(define (problem slice_bread_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    bread - object
    knife - object
    counterTop - object
  )
  (:init
    (at robot1 counterTop)
    (at-location bread counterTop)
    (at-location knife counterTop)
  )
  (:goal
    (and
      (sliced bread)
    )
  )
)