```pddl
(define (problem store-butterknife)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    butterKnife - object
    drawer1 - object
    counterTop - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)

    (at-location butterKnife counterTop)
    (at-location drawer1 floor)

    (object-close robot1 drawer1)
    (not (holding robot1 butterKnife))
  )

  (:goal (and
    (at-location butterKnife drawer1)
    (object-close robot1 drawer1)
  ))
)
```