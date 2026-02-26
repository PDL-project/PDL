(define (problem place-knife-on-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    knife - object
    countertop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 floor)
    (holding robot1 knife)
    (at-location countertop floor)
  )

  (:goal (and
    (at-location knife countertop)
    (holding robot1 knife)
  ))

  (:metric minimize (total-cost))
)