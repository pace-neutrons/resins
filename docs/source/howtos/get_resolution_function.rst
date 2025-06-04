How To Get the Resolution Function
==================================

The short answer for getting the :term:`resolution function` of a particular
:term:`version` of a given :term:`instrument` is to use the
:py:meth:`resolution_functions.instrument.Instrument.get_resolution_function`
method, e.g.:

>>> from resolution_functions import Instrument
>>> tosca = Instrument('TOSCA')
>>> book = tosca.get_resolution_function('book', detector_bank='Forward')
>>> print(book)
ToscaBookModel(citation="")

However, this is a shared interface, and so the
:term:`configurations<configuration>` and :term:`settings<setting>` that need to
be provided differ between :term:`instruments<instrument>`,
:term:`versions<version>`, and :term:`models<model>`. Of course, the information
required can be gathered from the documentation, and the :term:`configuration`
details can be found via :doc:`programmatic means<programmatic_query>`, but the
only way of obtaining all the information simultaneously is getting the model
signature:

How To Get the Model Signature
------------------------------

The
:py:meth:`resolution_functions.instrument.Instrument.get_model_signature`
method returns a :py:class:`~inspect.Signature` (from the `inspect` module) that
contains all the information for calling the
:py:meth:`resolution_functions.instrument.Instrument.get_resolution_function`
method:

* All arguments available for the :term:`model` & :term:`version` &
  :term:`instrument`
  * The type annotation for that argument
  * The default value for that argument
  * The restrictions on that argument

This method does not distinguish between :term:`configurations<configuration>`
and :term:`settings<setting>`; it only considers
:py:meth:`resolution_functions.instrument.Instrument.get_resolution_function`
and its call signature:

>>> from resolution_functions import Instrument
>>> maps = Instrument.from_default('MAPS')
>>> sig = maps.get_model_signature()
>>> sig
<Signature (model_name: Optional[str] = 'PyChop_fit', *, chopper_package: Literal['A', 'B', 'S'] = 'A', e_init: Annotated[ForwardRef('Optional[float]'), 'restriction=[0, 2000]'] = 500, chopper_frequency: Annotated[ForwardRef('Optional[int]'), 'restriction=[50, 601, 50]'] = 400, fitting_order: 'int' = 4, _) -> resolution_functions.models.pychop.PyChopModelFermi>

The returned :py:class:`inspect.Signature` object can be then be queried using
the full capabilities afforded by this standard library implementation. For
example, the return annotation and the arguments can be queried separately:

>>> sig.return_annotation
<class 'resolution_functions.models.pychop.PyChopModelFermi'>
>>> sig.parameters
mappingproxy(OrderedDict([('model_name', <Parameter "model_name: Optional[str] = 'PyChop_fit'">), ('chopper_package', <Parameter "chopper_package: Literal['A', 'B', 'S'] = 'A'">), ('e_init', <Parameter "e_init: Annotated[ForwardRef('Optional[float]'), 'restriction=[0, 2000]'] = 500">), ('chopper_frequency', <Parameter "chopper_frequency: Annotated[ForwardRef('Optional[int]'), 'restriction=[50, 601, 50]'] = 400">), ('fitting_order', <Parameter "fitting_order: 'int' = 4">), ('_', <Parameter "_">)]))

Where the latter can then be investigated in much more detail via the
:py:class:`inspect.Parameter` interface:

>>> sig.parameters['e_init']
<Parameter "e_init: Annotated[ForwardRef('Optional[float]'), 'restriction=[0, 2000]'] = 500">
>>> sig.parameters['e_init'].name
'e_init'
>>> sig.parameters['e_init'].default
500
>>> sig.parameters['e_init'].annotation
typing.Annotated[ForwardRef('Optional[float]'), 'restriction=[0, 2000]']
>>> sig.parameters['e_init'].kind
<_ParameterKind.KEYWORD_ONLY: 3>
